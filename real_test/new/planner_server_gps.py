#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
planner_gps.py + CSV Logger + Key Input
---------------------------------------
GPS 수신 + Redis 퍼블리시 + 로컬 CSV 로깅
키보드에서 's' 키 입력 시 CSV에 '-' 행 추가
"""

import os, time, json, threading, socket, base64, sys
from datetime import datetime
from queue import Queue
import redis
import serial
from pyubx2 import UBXReader, UBXMessage, SET
from pynmeagps.nmeamessage import NMEAMessage
import csv
import keyboard  # pip install keyboard

# ==============================
# 환경설정
# ==============================
SERIAL_PORT = os.getenv("GPS_SERIAL", "/dev/tty.usbmodem101")
SERIAL_BAUD = int(os.getenv("GPS_BAUD", "115200"))
REDIS_HOST  = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_KEY   = "gps_state"
CSV_PATH    = "gps_log.csv"

# ==============================
# 전역 변수
# ==============================
_gps_lat, _gps_lon = None, None
_gps_lock = threading.Lock()
_last_log = 0.0

# Redis 연결
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# CSV 파일 준비
csv_lock = threading.Lock()
csv_headers = ["timestamp", "lat", "lon"]
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)

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

def log_gps(lat, lon):
    """CSV에 GPS 데이터 기록"""
    with csv_lock, open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.utcnow().isoformat(), lat, lon])

def add_dash_row():
    """CSV에 '-' 행 추가"""
    with csv_lock, open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['-' for _ in csv_headers])
    print("[CSV] Added '-' row")

# ==============================
# 키 입력 스레드
# ==============================
def key_listener():
    print("[Key] Press 's' to add '-' row, 'q' to quit logging.")
    while True:
        if keyboard.is_pressed('s'):
            add_dash_row()
            time.sleep(0.4)  # 중복 방지
        elif keyboard.is_pressed('q'):
            print("[Key] Quit signal detected.")
            os._exit(0)
        time.sleep(0.1)

# ==============================
# GPS 수신 스레드
# ==============================
def gps_thread(ser):
    global _last_log
    ubr = UBXReader(ser, protfilter=7)
    print(f"[GPS] Listening on {ser.port}@{ser.baudrate}")

    while True:
        try:
            raw, msg = ubr.read()
            lat, lon = None, None

            # UBX NAV-PVT
            if isinstance(msg, UBXMessage) and msg.identity == 'NAV-PVT':
                lat = msg.lat * 1e-7
                lon = msg.lon * 1e-7

            # NMEA GGA
            elif isinstance(msg, NMEAMessage) and msg.identity.endswith("GGA"):
                lat = float(msg.lat)
                lon = float(msg.lon)

            if lat is not None and lon is not None:
                set_gps(lat, lon)
                publish_gps(lat, lon)
                log_gps(lat, lon)

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
# NTRIP Thread (Optional)
# ==============================
def ntrip_thread(caster, port, mountpoint, user, password, ser, init_queue):
    """RTCM 수신 + GGA 업링크"""
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
    threading.Thread(target=key_listener, daemon=True).start()

    # NTRIP 옵션
    caster     = os.getenv('caster')
    port       = os.getenv('port')
    mountpoint = os.getenv('mountpoint')
    user       = os.getenv('user')
    password   = os.getenv('password')

    if all([caster, port, mountpoint, user, password]):
        q = Queue(maxsize=2)
        q.put((37.0, 127.0))
        threading.Thread(
            target=ntrip_thread,
            args=(caster, port, mountpoint, user, password, ser, q),
            daemon=True
        ).start()
    else:
        print("[Main] NTRIP not configured; running standalone GPS.")

if __name__ == "__main__":
    print("[Server] Starting GPS → Redis + CSV logger")
    start_gps()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[Server] Stopped.")
