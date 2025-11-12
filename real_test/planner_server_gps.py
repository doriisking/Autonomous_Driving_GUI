#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner_gps.py
----------------
단독 GPS 수신 모듈 (UBX + NMEA 통합) → Redis JSON publish

- UBlox GNSS 수신기에서 lat/lon 데이터를 실시간으로 읽어 Redis에 저장
- 다른 프로세스(web_server.py 등)에서 redis.get("gps_state")로 접근 가능
- RTK(NTRIP) 옵션 포함 (환경변수로 caster/user/password 지정 시 자동)

환경변수:
    GPS_SERIAL=/dev/tty.usbmodem101
    GPS_BAUD=115200
    REDIS_HOST=localhost
    REDIS_PORT=6379
    caster, port, mountpoint, user, password (NTRIP 옵션)
"""

import os, time, json, threading, socket, base64
from datetime import datetime
from queue import Queue
import redis
import serial
from pyubx2 import UBXReader, UBXMessage, SET
from pynmeagps.nmeamessage import NMEAMessage

# ==============================
# 환경설정
# ==============================
SERIAL_PORT = os.getenv("GPS_SERIAL", "/dev/tty.usbmodem101")
SERIAL_BAUD = int(os.getenv("GPS_BAUD", "115200"))
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY   = "gps_state"

# ==============================
# 전역 변수
# ==============================
_gps_lat, _gps_lon = None, None
_gps_lock = threading.Lock()
_last_log = 0.0

# Redis 연결
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ==============================
# 헬퍼 함수
# ==============================
def set_gps(lat, lon):
    global _gps_lat, _gps_lon
    with _gps_lock:
        _gps_lat, _gps_lon = lat, lon

def get_gps():
    with _gps_lock:
        return _gps_lat, _gps_lon

def publish_gps(lat, lon):
    """Redis에 현재 lat/lon 상태 전송"""
    try:
        data = {
            "gps": {"lat": lat, "lon": lon},
            "timestamp": time.time()
        }
        r.set(REDIS_KEY, json.dumps(data))
    except Exception as e:
        print(f"[Redis] Publish error: {e}")

def make_gga(lat: float, lon: float) -> bytes:
    """위경도를 받아 NMEA GGA 문장(ASCII byte) 생성 (1 Hz 업링크용)."""
    t = datetime.utcnow().strftime("%H%M%S.00")
    lat_d = int(abs(lat)); lat_m = (abs(lat) - lat_d) * 60; lat_dir = 'N' if lat >= 0 else 'S'
    lon_d = int(abs(lon)); lon_m = (abs(lon) - lon_d) * 60; lon_dir = 'E' if lon >= 0 else 'W'
    core = (f"GPGGA,{t},{lat_d:02d}{lat_m:07.4f},{lat_dir},"
            f"{lon_d:03d}{lon_m:07.4f},{lon_dir},1,12,1.0,0.0,M,0.0,M,,")
    chk = 0
    for c in core:
        chk ^= ord(c)
    return f"${core}*{chk:02X}\r\n".encode('ascii')

# ==============================
# GPS 수신 (UBX + NMEA 통합 스레드)
# ==============================
def gps_thread(ser):
    """UBlox 수신기에서 UBX/NMEA 통합 처리"""
    global _last_log
    ubr = UBXReader(ser, protfilter=7)  # 1=UBX, 2=NMEA, 4=RTCM → 7이면 둘 다 허용
    print(f"[GPS] Listening on {ser.port}@{ser.baudrate}")

    while True:
        try:
            raw, msg = ubr.read()

            # UBX NAV-PVT (고정밀)
            if isinstance(msg, UBXMessage) and msg.identity == 'NAV-PVT':
                lat = msg.lat * 1e-7
                lon = msg.lon * 1e-7
                set_gps(lat, lon)
                publish_gps(lat, lon)

            # NMEA GGA (보조용)
            elif isinstance(msg, NMEAMessage) and msg.identity.endswith("GGA"):
                lat = float(msg.lat)
                lon = float(msg.lon)
                set_gps(lat, lon)
                publish_gps(lat, lon)

            # 로그 출력 (1Hz)
            now = time.time()
            if now - _last_log > 1.0:
                lat, lon = get_gps()
                if lat and lon:
                    print(f"[GPS] Lat={lat:.7f}, Lon={lon:.7f}")
                _last_log = now

        except serial.SerialException as e:
            print(f"[GPS] Serial Error: {e}")
            time.sleep(1.0)
        except Exception as e:
            print(f"[GPS] Error: {e}")
            time.sleep(0.5)

# ==============================
# NTRIP Client (옵션)
# ==============================
def ntrip_thread(caster, port, mountpoint, user, password, ser, init_queue):
    """NTRIP → RTCM 수신 → 시리얼로 전송, 1 Hz로 최신 GGA 업링크"""
    auth = base64.b64encode(f"{user}:{password}".encode()).decode()
    req = (
        f"GET /{mountpoint} HTTP/1.1\r\n"
        f"Host: {caster}:{port}\r\n"
        "User-Agent: NTRIP PythonClient/1.0\r\n"
        "Accept: */*\r\n"
        "Connection: keep-alive\r\n"
        f"Authorization: Basic {auth}\r\n\r\n"
    ).encode('ascii')

    lat, lon = init_queue.get(block=True)
    print(f"[NTRIP] Init position: {lat:.7f}, {lon:.7f}")

    while True:
        try:
            sock = socket.create_connection((caster, int(port)), timeout=10)
            sock.sendall(req)
            buf = b""
            while b"\r\n\r\n" not in buf:
                buf += sock.recv(1)
            sock.settimeout(1.0)
            last_gga = 0.0

            print("[NTRIP] RTCM stream started")
            while True:
                now = time.time()
                if now - last_gga >= 1.0:
                    lat_l, lon_l = get_gps()
                    if lat_l is not None:
                        lat, lon = lat_l, lon_l
                    sock.sendall(make_gga(lat, lon))
                    last_gga = now

                try:
                    chunk = sock.recv(1024)
                    if chunk:
                        ser.write(chunk)
                except socket.timeout:
                    pass
        except Exception as e:
            print(f"[NTRIP] Error: {e}")
            time.sleep(5)

# ==============================
# Main Entry
# ==============================
def start_gps():
    ser = serial.Serial(SERIAL_PORT, baudrate=SERIAL_BAUD, timeout=1)
    print(f"[Main] Serial opened: {SERIAL_PORT}@{SERIAL_BAUD}")

    threading.Thread(target=gps_thread, args=(ser,), daemon=True).start()

    # NTRIP 옵션
    caster     = os.getenv('caster')
    port       = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user       = os.getenv('user')
    password   = os.getenv('password')

    if all([caster, port, mountpoint, user, password]):
        q = Queue(maxsize=2)
        q.put((37.0, 127.0))  # 초기 dummy 좌표
        threading.Thread(
            target=ntrip_thread,
            args=(caster, port, mountpoint, user, password, ser, q),
            daemon=True
        ).start()
    else:
        print("[Main] NTRIP not configured; running standalone GPS.")

if __name__ == "__main__":
    print("[Server] Starting GPS → Redis publisher")
    start_gps()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Server] Stopped.")
