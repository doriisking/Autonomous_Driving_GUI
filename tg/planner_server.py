#!/usr/bin/env python3

from __future__ import annotations
import argparse, asyncio, csv, json, random, time
from typing import List, Tuple
import math

from datetime import datetime
import socket, threading
from multiprocessing import Process, set_start_method
from queue import Queue

import struct

import os, sys, glob, ctypes

# server
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path

import serial

# GPS
from pyubx2 import UBXReader, UBXMessage, SET

# GPS visualization
import folium
import argparse

from termcolor import colored

from dataclasses import dataclass

# camera streaming
import cv2

# custom code
from controller import PriorityPD
from nav_utils import *

#import periph.d435i
import periph.go2
import periph.gps
import periph.compute

# ---------------------------------
# Live-GPS shared state
# ---------------------------------
key             = 0
flag            = 0
buff            = {}
yaw_offset      = 0.0
waypoints_gps   = torch.zeros(6,2)
ctrl_msg        = ''
planner_msg     = ''

pub_flag = [True, True]

# ---------------------------------
# state variable
# ---------------------------------
selected_path_file = None
ALL_PATHS = {}

MISSION_ACTIVE = False
REACH_GOAL = False
INIT_GPS = False
UNDER_CALIBRATION = False

REAL_CONTROL = True
IS_NOT_STOPPED = False

queue_in, queue_out = Queue(), Queue()


# debug code segment for planner
DEBUG = False
DEBUG = True
# ---------------------------------
# control variable
# ---------------------------------
goal_dx, goal_dy = None, None
vigoal_dx, vigoal_dy = None, None

REACH_TOL = 3.0 # threshold distance for reaching goal (meter)

GOAL_RATE = 1.0 # goal position update rate
CTRL_RATE = 0.8

VIPLANNER_OFFSET_Y = 0.0

TRAJ_RATE = 0.9

CAM_HEIGHT      = +0.0 # meter
CAM_TILT_PITCH  = +3.0 # degree
CAM_TILT_YAW    = -2.0

VIS_TYPE = 'rgb(trajectory) super(rgb overlay) | seg depth'
VIS_TYPE = 'rgb super-green(rgb trajectory overlay)'

# ---------------------------------
# Initialize segmentator
# ---------------------------------
USE_SEGMENTATION = True
current_frame = None

def print_info(headline, text, color='green'):
    print(colored(headline + "\t", color), text)

def ema_update(accum, new, rate):
    return (1-rate) * accum + (rate) * new

def gps_to_xy(lat_curr, lon_curr,
               lat_goal, lon_goal,
               heading_deg):
    """GPS ↦ 평면 (x,y) [m]; x=heading 방향, y=좌측 +"""
    # 1) 동-북 오프셋
    dlat  = math.radians(lat_goal - lat_curr)
    dlon  = math.radians(lon_goal - lon_curr)
    lat_avg = math.radians((lat_goal + lat_curr) * 0.5)

    north = EARTH_R * dlat                       # N (m)
    east  = EARTH_R * math.cos(lat_avg) * dlon   # E (m)

    # 2) heading 기준 회전
    psi = math.radians(heading_deg)
    x =  east * math.sin(psi) + north * math.cos(psi)
    y = -east * math.cos(psi) + north * math.sin(psi)
    return x, y


def xy_to_gps(lat_curr, lon_curr, dxy, heading_deg, use_avg_lat=False):
    """
    평면 (x,y)[m] → GPS (lat, lon)
    goal_to_xy 함수의 역변환 (벡터화 버전)

    Args:
        lat_curr (float): 현재 위도 (deg)
        lon_curr (float): 현재 경도 (deg)
        dxy (torch.Tensor): (N, 2) tensor [dx, dy] (m)
        heading_deg (float): 현재 heading (deg, 북 기준)
        use_avg_lat (bool): 평균 위도로 1회 보정 여부

    Returns:
        gps_goal (torch.Tensor): (N, 2) tensor [[lat, lon], ...] (deg)
    """
    assert dxy.ndim == 2 and dxy.shape[1] == 2, "dxy must be (N, 2) tensor"
    device = dxy.device

    psi = math.radians(heading_deg)
    s, c = math.sin(psi), math.cos(psi)

    dx, dy = dxy[:, 0].double(), dxy[:, 1].double()

    # (x, y) → (east, north)
    east  =  s * dx - c * dy
    north =  c * dx + s * dy
    
    # EN → Δlat, Δlon
    lat_rad = math.radians(lat_curr)
    dlat = north / EARTH_R
    dlon = east  / (EARTH_R * math.cos(lat_rad))

    lat_goal = lat_curr + torch.rad2deg(torch.tensor(dlat, device=device).double())
    lon_goal = lon_curr + torch.rad2deg(torch.tensor(dlon, device=device).double())

    if use_avg_lat:
        lat_avg_rad = torch.deg2rad((lat_goal + lat_curr) * 0.5)
        dlon_refine = east / (EARTH_R * torch.cos(lat_avg_rad))
        lon_goal = lon_curr + torch.rad2deg(torch.tensor(dlon_refine, device=device))

    # (N, 2) tensor 반환
    gps_goal = torch.stack([lat_goal, lon_goal], dim=1)
    return gps_goal


def compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1):
    # 4) odom forward vector   (expected ≈ (distance, 0))
    odom_dx = x1 - x0
    odom_dy = y1 - y0
    yaw_from_odom = math.atan2(odom_dy, odom_dx)    # rad in odom frame

    # 5) global forward vector from GPS
    east, north  = haversine_xy(lat0, lon0, lat1, lon1)
    yaw_global   = math.atan2(east, north)          # rad, 0=north, CCW=left

    # 6) offset such that   yaw_global = yaw_from_odom + offset
    yaw_offset = math.degrees(normalize(yaw_global - yaw_from_odom))

    return yaw_offset


def calibrate_heading_gps(sport_client,
                          distance=2.0,
                          v_init=0.5,
                          sample_rate=20.0):
    """
    Drive +x 'distance' metres, compute yaw_offset = heading_global - yaw_odom.
    Returns yaw_offset [rad].
    """
    global planner_msg 
    global yaw_offset
    global IS_NOT_STOPPED
    global UNDER_CALIBRATION

    UNDER_CALIBRATION = True

    # 1) snapshot starting state
    x0, y0, yaw0 = periph.go2.pos[0], periph.go2.pos[1], periph.go2.angle[2]
    lat0, lon0   = periph.gps.lat, periph.gps.lon

    # 2) command forward motion
    t_start = time.time()
    loop_dt = 1.0 / sample_rate

    while True:
        x, y = periph.go2.pos[0], periph.go2.pos[1]
        dx = x - x0
        dy = y - y0
        travelled = math.hypot(dx, dy)

        if travelled >= distance:
            break
        if time.time() - t_start > 10:   # safety timeout
            break

        IS_NOT_STOPPED = True
        sport_client.Move(v_init, 0.0, 0.0)

        time.sleep(loop_dt)

    # stop
    sport_client.Move(0.0, 0.0, 0.0)
    IS_NOT_STOPPED = False

    # 3) snapshot ending pose
    x1, y1, yaw1 = periph.go2.pos[0], periph.go2.pos[1], periph.go2.angle[2]
    lat1, lon1   = periph.gps.lat, periph.gps.lon
   
    yaw_offset = compute_yaw_offset(x0,y0,yaw0,lat0,lon0,x1,y1,yaw1,lat1,lon1)

    planner_msg =f"offset (deg)        : {yaw_offset:.2f}"

    print_info("Calibration", planner_msg)

    UNDER_CALIBRATION = False


def smooth_stop():
    global IS_NOT_STOPPED, REAL_CONTROL
    if REAL_CONTROL:
        periph.go2.sport_client.Move(0,0,0)
        #sport_client.BalanceStand()
        #sport_client.SwitchGait(0)
    IS_NOT_STOPPED = False


def control_thread(sport_client, rate=10.0):
    global vigoal_dx, vigoal_dy
    global IS_NOT_STOPPED
    global INIT_GPS
    global REAL_CONTROL
    global ctrl_msg

    ctrl = PriorityPD()
    
    print_info("Control", 'start control thread')

    if periph.go2.sport_client is None:
        print('periph.go2.sport_client is None.')

    while not INIT_GPS:
        if isinstance(periph.gps.lon, float) and isinstance(periph.gps.lat, float):
            print_info("Control", f'GPS initialized {periph.gps.lon}, {periph.gps.lat}')
            INIT_GPS = True
        else:
            ctrl_msg = 'waiting for gps signal...'
            print_info("Control", ctrl_msg)
            time.sleep(5.0)

    while True:
        if MISSION_ACTIVE:
            break
        else:
            time.sleep(1.0)

    vx, vy, vyaw = 0.0, 0.0, 0.0
    while True:
        #if MISSION_ACTIVE and (goal_x is not None) and (goal_y is not None):
        if MISSION_ACTIVE and (vigoal_dx is not None) and (vigoal_dy is not None):
            if REACH_GOAL:
                #smooth_stop()
                periph.go2.sport_client.StopMove()
                IS_NOT_STOPPED = False
            else:
                #x0, y0 = periph.go2.pos[0], periph.go2.pos[1]
                #dx, dy = goal_x - x0 , goal_y - y0
                #vx_new, vy_new, vyaw_new = ctrl.step(dx, dy)
                vx_new, vy_new, vyaw_new = ctrl.step(vigoal_dx, vigoal_dy)
                new_ctrl_msg = ''
                if REAL_CONTROL:
                    vx, vy, vyaw = ema_update(vx, vx_new, CTRL_RATE), ema_update(vy, vy_new, CTRL_RATE), ema_update(vyaw, vyaw_new, CTRL_RATE)
                    periph.go2.sport_client.Move(vx,vy,vyaw)
                    IS_NOT_STOPPED = True
                    new_ctrl_msg += 'Real-'
                new_ctrl_msg += f'Move({vx:.1f},{vy:.1f},{vyaw:.1f})'
                ctrl_msg = new_ctrl_msg
                #print_info("Control", ctrl_msg)
                time.sleep(1/rate)
        else:
            if periph.go2.sport_client is not None:
                if IS_NOT_STOPPED:
                    #smooth_stop()
                    periph.go2.sport_client.StopMove()
                    IS_NOT_STOPPED = False


def planner_thread(queue_in, queue_out, rate=1.0):
    global goal_dx, goal_dy, vigoal_dx, vigoal_dy
    global yaw_offset
    global waypoints_gps
    global planner_msg # the others
    global GOAL_RATE
    global seg_image

    global MISSION_ACTIVE, INIT_GPS, REACH_GOAL, REACH_TOL

    f = None
    print_info("Planner", "queue_out wait for segmentation init.")
    queue_out.join() # wait for model initialization
    
    #goal_x, goal_y = periph.go2.pos[0], periph.go2.pos[1]
    goal_dx, goal_dy, goal_dz = 0.0, 0.0, 0.3 

    while DEBUG:
        #dx, dy = 10.0, 0
        #goal_dxyz = [5.0, 0.0, 0.3]
        goal_dxyz = [5.0, 0.0, 0.0] # dx-forward, dy-right, dz-up
        x0, y0 = periph.go2.pos[0], periph.go2.pos[1]

        lat0, lon0, hdop0 = periph.gps.lat, periph.gps.lon, periph.gps.gps_hdop
        body0 = periph.go2.body_height
        angle0 = periph.go2.angle

        #height_go2 = 0.20
        delay_t, pre_time, t0 = time.time(), time.time(), time.time()
       
        #msg_in = (goal_dxyz, [periph.go2.body_height, periph.go2.angle[0], periph.go2.angle[1]]) # pass local goal to planner
        msg_in = (goal_dxyz, [x0, y0, body0], angle0, [lat0, lon0]) # pass local goal to planner


        #msg_in = ([goal_dx, goal_dy], [periph.go2.body_height, 0.0, 0.0]) # pass local goal to planner

        queue_in.put(msg_in)
        #print_info("Planner", f"put goal ({dx},{dy})")

        trajs, waypoints, fear = queue_out.get() # block until single element
        #print_info("Planner", 'get seg+waypoints.')
        dt = time.time() - t0
        #goal_x, goal_y = float(waypoints[0, 0, 0]) + x0, float(waypoints[0, 0, 1]) + y0
        print(waypoints)
        vigoal_dx, vigoal_dy = float(waypoints[0, 0]), float(waypoints[0, 1])

        lat_cur, lon_cur = periph.gps.lat, periph.gps.lon
        if type(lat_cur) != float or type(lon_cur) != float:
            lat_cur, lon_cur = 37.566500, 126.978010
       
        waypoints = torch.cat((torch.Tensor(waypoints), torch.Tensor([goal_dxyz])),dim=0)
        waypoints_gps = xy_to_gps(lat_cur, lon_cur, waypoints[:, 0:2], periph.go2.angle[2] + yaw_offset)

        print(waypoints)
        #print_info("Planner", waypoints[0, :, 0:2])
        t0 = time.time()

    t0 = time.time()
    while not INIT_GPS:
        if time.time() - t0 > 5.0:
            print_info("GPS", "Wait for gps initialization")
            t0 = time.time()
        time.sleep(1/rate)

    while True:
        if MISSION_ACTIVE:
            print_info("Planner", 'start new path')
            path = LinearPath(ALL_PATHS[selected_path_file]['coords'], reach_tol=REACH_TOL) # 1.0 m for pedestrian mode
            REACH_GOAL = False
            lat_pre, lon_pre = periph.gps.lat, periph.gps.lon
            x_pre, y_pre = periph.go2.pos[0], periph.go2.pos[1]
            yaw_pre = periph.go2.angle[2]
            #goal_x, goal_y = periph.go2.pos[0], periph.go2.pos[1]
            goal_dx, goal_dy = 0.0, 0.0
            t_pre = time.time()

            # 현재 시간 불러오기
            now = datetime.now()
            # 파일 이름을 "YYYYMMDD_HHMMSS.txt" 형태로 지정
            filename = now.strftime("logs/%Y%m%d_%H%M%S.log")

            # 파일 작성
            f = open(filename, "w")
            f.write('x0,y0,roll,pitch,yaw,yaw_offset,lat,lon,hdop\n')

            while not REACH_GOAL and MISSION_ACTIVE:
                goal, is_goal_updated = path.update(periph.gps.lat, periph.gps.lon)
                global_heading = math.degrees(normalize(math.radians(periph.go2.angle[2] + yaw_offset)))
                #print_info("Planner", 'not reach goal', goal, is_goal_updated, global_heading)
                if goal is None:
                    REACH_GOAL = True
                    MISSION_ACTIVE = False
                   
                    planner_msg = f'Path Finished!'
                    print_info("Planner", 'PATH FINISH', 'red')
                    try:
                        f.close()
                    except:
                        print_info("Log", "failed to close log file.")
                else:
                    lat0, lon0, hdop0 = periph.gps.lat, periph.gps.lon, periph.gps.gps_hdop
                    x0, y0 = periph.go2.pos[0], periph.go2.pos[1]
                    body0 = periph.go2.body_height
                    angle0 = periph.go2.angle
                    yaw0 = angle0[2]
                    
                    # update yaw_offset when distance is more than 2-meters
                    if (x_pre - x0) ** 2 + (y_pre - y0) ** 2 > 4.0:
                        t_cur = time.time()
                        yaw_offset_new = compute_yaw_offset(x_pre,y_pre,yaw_pre,lat_pre,lon_pre,x0,y0,yaw0,lat0,lon0)
                        yaw_offset = yaw_offset * 0.9 + yaw_offset_new * 0.1
                        planner_msg = 'update yaw_offset:{yaw_offset:.2f} yaw_off_new:{yaw_offset_new:.2f}'
                        # update values
                        x_pre, y_pre = x0, y0
                        lat_pre, lon_pre = lat0, lon0

                    t_cur = time.time()
                    if (t_cur - t_pre) >= 0.2: # 5 hz
                        f.write(f'{x0},{y0},{angle0[0]},{angle0[1]},{yaw0},{yaw_offset},{lat0},{lon0},{hdop0}\n')
                        t_pre = t_cur

                    dx, dy = gps_to_xy(lat0, lon0, goal[0], goal[1], global_heading)
                    dz = 0.0

                    # viplanner
                    msg_in = ([dx, dy, dz], [x0, y0, body0], angle0, [lat0, lon0]) # pass local goal to planner
                    queue_in.put(msg_in)
                    #print_info("Planner", "put rgb+depth.")

                    trajs, waypoints, fear = queue_out.get() # block until single element
                    #print_info("Planner", 'get seg+waypoints.')
                  
                    vigoal_dx, vigoal_dy =  float(waypoints[0, 0]), float(waypoints[0, 1])

                    #planner_msg = f'sub-goal [{path.idx}/{len(path.waypoints)}] gps (dx,dy)=({dx:.1f},{dy:.1f}) viplanner (dx,dy)=({vigoal_x:.1f},{vigoal_y:.1f}) '
                    planner_msg = f'sub-goal [{path.idx}/{len(path.waypoints)}] gps (dx,dy)=({dx:.1f},{dy:.1f}) viplanner (dx,dy)=({vigoal_dx:.1f},{vigoal_dy:.1f}) '

                    waypoints = torch.cat((torch.Tensor(waypoints[:, 0:2]), torch.Tensor([[dx, dy]])),dim=0)
                    waypoints_gps = xy_to_gps(lat0, lon0, waypoints , periph.go2.angle[2] + yaw_offset)

        else:
            planner_msg = 'Switch to inactive mission.'
            REACH_GOAL = False
            goal_dx, goal_dy = 0.0, 0.0
            if f is not None:
                try:
                    f.close()
                except:
                    print_info("Log", "failed to close log file.")
       
            time.sleep(1/rate)


def main_thread(CAMERA_TYPE):
    global REAL_CONTROL
    global queue_in, queue_out
    # --------------------------------
    # model configuration
    # --------------------------------
    EXTENSION_DOCK = 'jetson' # or 'go2-edu'
    CONTROL_RATE = 10.0
   
    # Serial information for GPS
    GPS_TYPE    = 'ublox-f9p'
    gps_port    = '/dev/ttyACM0'
    gps_baud    = 115200

    # NTRIP server information
    caster_host = 'rts2.ngii.go.kr'
    caster_port = 2101
    mountpt     = 'VRS-RTCM32'
    user        = 'tackgeun90'
    password    = 'ngii'

    #caster_host = 'rts1.ngii.go.kr'
    #caster_port = 2101
    #mountpt     = 'VRS-RTCM34'
    #user        = 'tackgeun90'
    #password    = 'ngii'



    # GPS Thread
    latlon_queue: Queue[Tuple[float, float]] = Queue(maxsize=1)
    ser = serial.Serial(gps_port, gps_baud, timeout=1.0)

    if GPS_TYPE == 'ublox-f9p':
        ser.write(UBXMessage('CFG','CFG-RATE', SET, measRate=100, navRate=1, timeref=0).serialize())
        time.sleep(0.1)
        ser.write(UBXMessage('CFG','CFG-MSG', SET, msgClass=0x01, msgID=0x07, rateUART1=1).serialize())
        time.sleep(0.1)

    # ntrip client for rtk correction
    # NTRIP > RTCM -> Serial thread
    ntrip_th = threading.Thread(
        target=periph.gps.ntrip_thread,
        args=(caster_host, caster_port, mountpt, user, password, ser, latlon_queue),
        daemon=True
    )
    ntrip_th.start()

    # ublox gps sensor thread
    # Serial -> UBX 파싱 -> 위치 출력 thread
    ubx_th = threading.Thread(
        target=periph.gps.ubx_thread,
        args=(ser, latlon_queue),
        daemon=True
    )
    ubx_th.start()

    # Owl4 camera and depth thread
    config = {
        'traj_rate': TRAJ_RATE,
        'cam_height': CAM_HEIGHT,
        'cam_tilt_pitch': CAM_TILT_PITCH,
        'cam_tilt_yaw': CAM_TILT_YAW,
        'vis_type': VIS_TYPE,
        'viplanner_offset_y': VIPLANNER_OFFSET_Y
    }
  
    segment_th = threading.Thread(
            target=periph.compute.compute_thread,
            args=(queue_in, queue_out, config, CAMERA_TYPE),
            daemon=True
    )
    segment_th.start()



    REAL_CONTROL = periph.go2.init_sport_client(REAL_CONTROL, EXTENSION_DOCK)

    control_th = threading.Thread(
            target=control_thread,
            args=(periph.go2.sport_client, CONTROL_RATE),
            daemon=True
    )
    control_th.start()

    queue_out.put('pop after segmentator initialization.')

    planner_th = threading.Thread(
            target=planner_thread,
            args=(queue_in, queue_out, CONTROL_RATE,),
            daemon=True
    )
    planner_th.start()


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

ALL_PATHS = load_all_paths()
selected_path_file = list(ALL_PATHS.keys())[0] if ALL_PATHS else None

# API endpoint
@app.get("/", response_class=HTMLResponse)
async def index():
    return open("planner_view.html", "r", encoding="utf-8").read()

@app.get("/paths", response_class=JSONResponse)
async def paths():
    return list(ALL_PATHS.values())

@app.get("/set_path")
async def set_path(file: str):
    global selected_path_file
    if file in ALL_PATHS:
        selected_path_file = file
        return {"success": True, "selected": file}
    return {"success": False, "error": "File not found"}

@app.get("/selected_path", response_class=JSONResponse)
async def selected_path():
    if selected_path_file and selected_path_file in ALL_PATHS:
        return ALL_PATHS[selected_path_file]
    return {"coords": [], "file": None}

@app.get("/toggle_mission")
async def toggle_mission():
    global MISSION_ACTIVE
    if periph.go2.sport_client is not None:
        MISSION_ACTIVE = not MISSION_ACTIVE
    else:
        print_info("Go2", "Sport Client is None")
    return {"active": MISSION_ACTIVE}

@app.get("/toggle_control")
async def toggle_mission():
    global REAL_CONTROL
    REAL_CONTROL = not REAL_CONTROL
    if REAL_CONTROL:
        print_info("Server", 'change to real control')
    else:
        print_info("Server", 'freeze mode')
    return {"active": REAL_CONTROL}

@app.get("/calibrate_heading")
async def trigger_calibration():
    print_info("SERVER", "trigger_calibration")
    try:
        calibrate_heading_gps(periph.go2.sport_client)
    except:
        print_info("Calibration", "Failed to calibrate properly")
    return {"active": True}

@app.get("/mission_status")
async def mission_status():
    return {"active": MISSION_ACTIVE}

@app.get("/control_status")
async def control_status():
    return {"active": REAL_CONTROL}

@app.get("/gps")
async def gps_stream():
    async def event_gen():
        while True:
            await asyncio.sleep(0.1)
            payload = {
                "lat": periph.gps.lat,
                "lon": periph.gps.lon,
                "hdop": periph.gps.gps_hdop,
                "lat0": float(waypoints_gps[0,0]),
                "lon0": float(waypoints_gps[0,1]),
                "lat2": float(waypoints_gps[2,0]),
                "lon2": float(waypoints_gps[2,1]),
                "lat5": float(waypoints_gps[5,0]),
                "lon5": float(waypoints_gps[5,1]),               
                "heading": periph.go2.angle[2] + yaw_offset,
                "heading_odom": periph.go2.angle[2],
                "status": periph.go2.status,
                "planner": f"[PLAN]{planner_msg}",
                "command": f"[CTRL]{ctrl_msg}"
            }
            yield f"data: {json.dumps(payload)}\n\n"
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/camera")
async def camera_stream(fps: int = 3, fmt: str = "webp", q: int = 75):
    """
    multipart/x-mixed-replace stream
    ex) <img src="/camera?fps=10&fmt=webp&q=60">
        fmt: png|jpg|webp
        q  : 품질 (webp/jpg: 1~100, png: 0~9)
    """
    fps = max(1, min(fps, 30))
    interval = 1.0 / fps

    # 필요시 스케일 결정 (기존 코드 유지)
    if VIS_TYPE == 'rgb(trajectory) super(rgb overlay) | seg depth':
        x_scale, y_scale = 4, 4
    elif VIS_TYPE == 'rgb super-green(rgb trajectory overlay)':
        x_scale, y_scale = 4, 8
    else:
        x_scale, y_scale = 4, 4

    # 포맷/헤더
    fmt = fmt.lower()
    if fmt not in ("png", "jpg", "jpeg", "webp"):
        fmt = "webp"
    ext = ".png" if fmt == "png" else (".jpg" if fmt in ("jpg","jpeg") else ".webp")
    ctype = b"image/png" if fmt == "png" else (b"image/jpeg" if fmt in ("jpg","jpeg") else b"image/webp")

    # 품질 파라미터
    enc_params = []
    if fmt == "png":
        level = min(9, max(0, q if 0 <= q <= 9 else 9))
        enc_params = [cv2.IMWRITE_PNG_COMPRESSION, level]
    elif fmt in ("jpg", "jpeg"):
        enc_params = [cv2.IMWRITE_JPEG_QUALITY, min(100, max(1, q))]
    else:  # webp
        enc_params = [cv2.IMWRITE_WEBP_QUALITY, min(100, max(1, q))]

    textPos = (0, 45*y_scale//2+10)
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.45
    color = (0, 255, 0)
    thickness = 1
    lineType = cv2.LINE_AA

    async def gen():
        boundary = b'--frame\r\n'
        while True:
            frm = None if periph.compute.current_frame is None else periph.compute.current_frame.copy()
            if frm is None:
                await asyncio.sleep(0.1)
                continue

            frm = cv2.resize(frm, dsize=(80*x_scale, 45*y_scale), interpolation=cv2.INTER_AREA)

            if UNDER_CALIBRATION:
                cv2.putText(frm, "DIRECTION CALIBRATION", textPos, fontFace, fontScale, color, thickness, lineType)
            #elif (not DEBUG) and (not (MISSION_ACTIVE and REAL_CONTROL)):
            elif (not DEBUG) and (not (MISSION_ACTIVE)):
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            elif DEBUG:
                cv2.putText(frm, "DEBUG", textPos, fontFace, fontScale, color, thickness, lineType)
            else:
                cv2.putText(frm, "Autonomous Navigation", textPos, fontFace, fontScale, color, thickness, lineType)

            ok, buf = cv2.imencode(ext, frm, enc_params)
            if not ok:
                await asyncio.sleep(0.01)
                continue
            data = buf.tobytes()

            # 일부 클라이언트는 Content-Length가 있으면 더 안정적
            yield boundary
            yield b"Content-Type: " + ctype + b"\r\n"
            yield f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
            yield data + b"\r\n"

            await asyncio.sleep(interval)

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')
#
@app.on_event('startup')
def start_sensor():
    #CAMERA_TYPE = 'Owl-4'
    CAMERA_TYPE = 'd435i'
    main_thread(CAMERA_TYPE)
