#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
sys.path.append(os.path.dirname(__file__))
import redis
import cv2
import math
import time
import numpy as np
import rclpy
import csv
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker
from datetime import datetime
r = redis.Redis(host='127.0.0.1', port=6379, db=0)
latest_grid_msg = None 

# CUDA (옵션)
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda  # noqa: F401
from pycuda.compiler import SourceModule  # noqa: F401

# ---- 거리맵 함수 (같은 폴더의 distmap_def.py) ----
from distmap_def import (
    build_dist_map_bfs_cuda,      # CUDA BFS
    build_dist_map_bf_cuda,       # CUDA Brute-Force (이 이름 그대로 사용)
    distmap_to_occupancygrid,     # (옵션) 시각화용
)


class DWACommandNode(Node):
    """
    전방 창(window)에서 최소 코스트 셀을 골라 /cmd(Twist) 발행.
    - 코스트: (x-dx)^2 + (y-dy)^2 + [d<margin] * penalty * (1 - d/margin)^2
    - vx(+전진), vyaw(+좌회전) 생성
    - 좌표계: 로봇 기준 (+x: 전방, +y: 좌측)
    - 정지 조건: 전방 창(window)에 free space(occ==0)가 하나도 없으면 정지
      (unknown(-1)은 파라미터에 따라 free 또는 obstacle 취급)
    """

    def __init__(self):
        super().__init__("dwa_command_node")

        # -------------------- 기본/코스트 파라미터 --------------------
        self.declare_parameter("penalty", 10.0)            # 장애물 페널티 상수
        self.declare_parameter("margin", 1.0)              # 안전 여유[m]
        self.declare_parameter("dx", 0.0)                  # 상위(GPS)가 준 목표 x[m] (로봇 기준)
        self.declare_parameter("dy", 0.0)                  # 상위(GPS)가 준 목표 y[m] (로봇 기준)

        # -------------------- 검사 창(Window) --------------------
        self.declare_parameter("ahead_m", 2.5)             # 전방 길이[m]
        self.declare_parameter("half_width_m", 1.0)        # 좌우 반폭[m]
        self.declare_parameter("stride", 1)                # 셀 스킵 간격(샘플링)

        # unknown 처리 (초기 관측전 출발성 확보 위해 기본 False 권장)
        self.declare_parameter("unknown_is_obstacle", False)

        # -------------------- 속도 생성 파라미터 --------------------
        self.declare_parameter("kv", 0.6)                  # 거리→전진속도 게인
        self.declare_parameter("kyaw", 1.2)                # 각도→회전속도 게인
        self.declare_parameter("v_max", 0.7)               # 전진 최대[m/s]
        self.declare_parameter("w_max", 0.75)              # 회전 최대[rad/s]
        self.declare_parameter("v_min", 0.0)               # 전진 최소[m/s]

        # -------------------- 회전 우선 옵션 --------------------
        self.declare_parameter("safety_slowdown", True)    # d<margin 감속
        self.declare_parameter("enable_turn_in_place", True)
        self.declare_parameter("theta_turn_deg", 35.0)     # 큰 각도면 제자리 회전
        self.declare_parameter("allow_backward_target", False)

        # -------------------- 주기 --------------------
        self.declare_parameter("timer_dt", 0.1)            # 타이머 주기(초)

        # -------------------- 토픽 --------------------
        self.declare_parameter("occ_topic", "/bev/occupancy_grid")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("marker_topic", "/dwa/local_goal_marker")

        # ---- 거리맵 관련 (방식 토글 + 최대거리 + 시각화) ----
        self.declare_parameter("dist_method", "bfs_cuda")
        self.declare_parameter("dist_max_m", 3.0)          # 거리맵 최대 반경[m]
        self.declare_parameter("publish_distgrid", False)  # 거리맵을 OccGrid로 내보내기

        # ---- 파라미터 로드 ----
        self.penalty = float(self.get_parameter("penalty").value)
        self.margin  = float(self.get_parameter("margin").value)
        self.dx      = float(self.get_parameter("dx").value)
        self.dy      = float(self.get_parameter("dy").value)

        self.ahead_m      = float(self.get_parameter("ahead_m").value)
        self.half_width_m = float(self.get_parameter("half_width_m").value)
        self.stride       = int(self.get_parameter("stride").value)

        self.unknown_is_obstacle = bool(self.get_parameter("unknown_is_obstacle").value)

        self.kv    = float(self.get_parameter("kv").value)
        self.kyaw  = float(self.get_parameter("kyaw").value)
        self.v_max = float(self.get_parameter("v_max").value)
        self.w_max = float(self.get_parameter("w_max").value)
        self.v_min = float(self.get_parameter("v_min").value)

        self.slow           = bool(self.get_parameter("safety_slowdown").value)
        self.turn_mode      = bool(self.get_parameter("enable_turn_in_place").value)
        self.theta_turn     = math.radians(float(self.get_parameter("theta_turn_deg").value))
        self.allow_backward = bool(self.get_parameter("allow_backward_target").value)

        self.dt = float(self.get_parameter("timer_dt").value)

        self.occ_topic    = self.get_parameter("occ_topic").value
        self.cmd_topic    = self.get_parameter("cmd_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value

        self.dist_method  = str(self.get_parameter("dist_method").value).lower()
        self.dist_max_m   = float(self.get_parameter("dist_max_m").value)
        self.pub_dist_occ = None
        if bool(self.get_parameter("publish_distgrid").value):
            self.pub_dist_occ = self.create_publisher(OccupancyGrid, "/dwa/dist_grid", 10)

        # ---- 상태 ----
        self._occ  = None                  # OccupancyGrid data (int8 HxW)
        self._info = None                  # (res, W, H, x0, y0)
        self._dist = None                  # 거리맵 (float32 HxW) [m]
        self._vx_prev = 0.0
        self._wz_prev = 0.0
        self._t_prev  = time.time()
        self._last_log_t = 0.0             # 로그 rate limit용

        # 외부 /cmd_vel 패스스루 상태
        self._ext_cmd = None  # type: Twist | None

        # ---- I/O ----
        self.create_subscription(OccupancyGrid, self.occ_topic, self._cb_occ, 10)
        self.pub_cmd    = self.create_publisher(Twist, self.cmd_topic, 10)
        self.pub_marker = self.create_publisher(Marker, self.marker_topic, 10)
        self.sub_dxdy   = self.create_subscription(Point, "/dxdy", self._cb_dxdy, 10)
        self.sub_extcmd = self.create_subscription(Twist, "/cmd_vel", self._cb_cmd_vel, 10)
        

        self.timer = self.create_timer(self.dt, self._on_timer)

        # === CSV 로깅 설정 ===
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = f"dwa_log_{timestamp}.csv"  # 자동 시간 이름 생성
        self._log_fp = open(self._log_path, "w", newline="")
        self._log_writer = csv.writer(self._log_fp)
        self._log_writer.writerow([
            "t",
            "dx_gps","dy_gps",          # 구독으로 들어온 GPS 목표
            "dx_dwa","dy_dwa",          # DWA 선택 셀 좌표
            "vx_cmd","vyaw_cmd",        # 명령 속도
            "kv","kyaw",                # 게인
            "stop_reason"               # 정지 이유 (front_window_blocked / none)
        ])
        self._log_fp.flush()

        self.get_logger().info(
            f"[dwa_command_node] L={self.ahead_m}m, ±{self.half_width_m}m | "
            f"penalty={self.penalty}, margin={self.margin} | "
            f"kv={self.kv}, kyaw={self.kyaw}, vmax={self.v_max}, wmax={self.w_max} | "
            f"stride={self.stride}, dt={self.dt}s | TurnInPlace={self.turn_mode} | "
            f"dist_method={self.dist_method}"
        )

    # ------------------------- 콜백 -------------------------
    def _cb_dxdy(self, msg: Point):
        self.dx = float(msg.x)
        self.dy = float(msg.y)

    def _cb_cmd_vel(self, msg: Twist):
        self._ext_cmd = msg
   
    def listener_callback(self, msg):
        global latest_grid_msg
        latest_grid_msg = msg 
    
    def _cb_occ(self, msg: OccupancyGrid):
        H = int(msg.info.height)
        W = int(msg.info.width)
        self._occ = np.asarray(msg.data, dtype=np.int8).reshape(H, W)
        self._info = (
            float(msg.info.resolution),
            W, H,
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )

        # ---- 거리맵 생성 ----
        method = self.dist_method
        try:
            if method in ("bfs_cuda", "bfs", "cuda"):
                self._dist = build_dist_map_bfs_cuda(msg, max_dist=self.dist_max_m)
            elif method in ("bruteforce", "brute", "bf"):
                self._dist = build_dist_map_bf_cuda(msg, max_dist=self.dist_max_m)
            else:
                self._dist = None
        except Exception as e:
            self._dist = None
            if (time.time() - self._last_log_t) > 1.0:
                self._last_log_t = time.time()
                self.get_logger().warn(f"[distmap] build failed: {e}")

        # (옵션) 거리맵을 OccGrid로 내보내어 RViz에서 확인
        if self.pub_dist_occ is not None and self._dist is not None:
            dist_occ = distmap_to_occupancygrid(self._dist, msg, max_dist=self.dist_max_m)
            self.pub_dist_occ.publish(dist_occ)
        # -------------------------------
        # 💡 Redis로 전송 (FastAPI용 브릿지)
        # -------------------------------
        try:
            img = ((100 - np.clip(self._occ, 0, 100)) * 2.55).astype(np.uint8)
            ok, png = cv2.imencode(".png", img)
            if ok:
                r.set("occ_grid_latest", png.tobytes())
            if self._dist is not None:
                dist_norm = np.clip(self._dist / self.dist_max_m, 0, 1)
                dist_img = (255 * (1.0 - dist_norm)).astype(np.uint8)
                ok, png2 = cv2.imencode(".png", dist_img)
                if ok:
                    r.set("dist_map_latest", png2.tobytes())
        except Exception as e:
            self.get_logger().warn(f"[RedisBridge] failed: {e}")
    # ------------------------- 유틸 -------------------------
    def _window_fully_blocked(self, res: float, W: int, H: int, x0: float, y0: float,
                              j0: int, i0: int) -> bool:
        """
        로봇 기준 (x: 전방+, y: 좌+)에서
        x ∈ [0, ahead_m], y ∈ [-half_width_m, +half_width_m] 직사각형 창 내부에
        free(=0) 셀이 '단 하나도 없으면' True.
        unknown_is_obstacle=False면 -1도 통과로 간주.
        """
        if self._occ is None:
            return False

        j_start = max(0, j0)
        j_end   = min(W, j0 + int(self.ahead_m / res) + 1)
        i_start = max(0, i0 - int(self.half_width_m / res))
        i_end   = min(H, i0 + int(self.half_width_m / res) + 1)

        if j_start >= j_end or i_start >= i_end:
            return False  # 창이 유효하지 않으면 막힘 판정 안 함

        # stride 샘플링 적용
        step = max(1, int(self.stride))
        win = self._occ[i_start:i_end:step, j_start:j_end:step]

        # free(0)가 하나라도 있으면 '막히지 않음'
        if np.any(win == 0):
            return False

        # unknown 처리
        if not self.unknown_is_obstacle and np.any(win < 0):
            return False

        # free도 없고, unknown도 (옵션상) 장애물로 취급 → 완전 차단
        return True

    def _publish_stop(self, reason: str):
        cmd = Twist()  # 모두 0
        self.pub_cmd.publish(cmd)

        t_now = time.time()
        try:
            self._log_writer.writerow([
                float(t_now),
                float(self.dx), float(self.dy),
                float('nan'), float('nan'),
                0.0, 0.0,
                float(self.kv), float(self.kyaw),
                reason
            ])
            self._log_fp.flush()
        except Exception:
            pass

        if (t_now - self._last_log_t) > 0.3:
            self._last_log_t = t_now
            self.get_logger().warn(f"[STOP] {reason} -> cmd(0,0)")

    # ------------------------- 주기 처리 -------------------------
    def _on_timer(self):
        t_now = time.time()

        # 1) 외부 /cmd_vel 패스스루 (선택적)
        if self._ext_cmd is not None:
            # 특정 sentinel(예: wz=-10.0)로 패스스루 모드 트리거
            if abs(self._ext_cmd.angular.z - (-10.0)) < 1e-9:
                self.pub_cmd.publish(self._ext_cmd)
                if (t_now - self._last_log_t) > 0.3:
                    self._last_log_t = t_now
                    self.get_logger().info(
                        f"[passthrough] /cmd <- /cmd_vel (vx={self._ext_cmd.linear.x:.2f}, wz={self._ext_cmd.angular.z:.2f})"
                    )
                return

        # 2) 내부 DWA 계산 전: 전방 창 완전 차단 시 정지
        if self._occ is not None and self._info is not None:
            res, W, H, x0, y0 = self._info
            # 로봇(0,0)의 격자 인덱스 (j: x, i: y) — 맵이 로봇 좌표와 평행(회전0)이라고 가정
            j0 = int((0.0 - x0) / res)
            i0 = int((0.0 - y0) / res)
            if 0 <= j0 < W and 0 <= i0 < H:
                if self._window_fully_blocked(res, W, H, x0, y0, j0, i0):
                    self._publish_stop("front_window_blocked")
                    return

        # 3) 내부 DWA 계산
        if self._occ is None or self._info is None:
            return
        if self._dist is None:
            return  # 아직 거리맵 준비 안 됨

        dt = max(1e-3, t_now - self._t_prev)
        self._t_prev = t_now

        res, W, H, x0, y0 = self._info

        # 로봇(0,0)의 격자 인덱스 (i: y, j: x)
        j0 = int((0.0 - x0) / res)
        i0 = int((0.0 - y0) / res)

        # 전방 창 범위
        j_start = max(0, j0)
        j_end   = min(W, j0 + int(self.ahead_m / res) + 1)
        i_start = max(0, i0 - int(self.half_width_m / res))
        i_end   = min(H, i0 + int(self.half_width_m / res) + 1)
        if j_start >= j_end or i_start >= i_end:
            return

        # ------ 최소 코스트 셀 탐색 ------
        best = None  # (cost, i, j, x, y, d)
        m = max(1e-6, self.margin)
        step = max(1, self.stride)

        for i in range(i_start, i_end, step):
            y = i * res + y0
            base_y = (y - self.dy) ** 2
            for j in range(j_start, j_end, step):
                x = j * res + x0
                base = (x - self.dx) ** 2 + base_y
                d = float(self._dist[i, j])  # 장애물까지의 거리[m] (0~dist_max)
                obs = self.penalty * (1.0 - d / m) ** 2 if d < m else 0.0
                cost = base + obs
                if (best is None) or (cost < best[0]):
                    best = (cost, i, j, x, y, d)

        if best is None:
            return

        _, bi, bj, bx, by, bd = best
        dx_dwa, dy_dwa = bx, by  # 로컬 목표 (로봇 기준)

        # --- RViz Marker (로컬 목표) ---
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = "base_link"
        marker.ns = "dwa_local_goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(dx_dwa)
        marker.pose.position.y = float(dy_dwa)
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.2
        marker.color.r = 1.0; marker.color.g = 0.2; marker.color.b = 0.2; marker.color.a = 1.0
        self.pub_marker.publish(marker)

        # ------ 속도 생성 ------
        theta = math.atan2(dy_dwa, dx_dwa)   # +면 좌회전
        r = math.hypot(dx_dwa, dy_dwa)

        vx_raw = self.kv * r * math.cos(theta)
        wz_raw = self.kyaw * theta

        if not self.allow_backward and dx_dwa < 0.0:
            vx_raw = 0.0

        if self.turn_mode and abs(theta) > self.theta_turn:
            vx_raw = 0.0  # 제자리 회전

        if self.slow and bd < m:
            scale = max(0.0, min(1.0, bd / m))
            vx_raw *= scale

        # 포화
        vx_cmd = max(self.v_min, min(self.v_max, vx_raw))
        wz_cmd = max(-self.w_max, min(self.w_max, wz_raw))

        # 사용자가 원했던 vx 고정값 유지(정지 조건에서만 0으로 덮어씀)
        vx_cmd = 0.7

        # 퍼블리시
        cmd = Twist()
        cmd.linear.x  = float(vx_cmd)
        cmd.angular.z = float(wz_cmd)
        self.pub_cmd.publish(cmd)

        # === CSV 로깅 ===
        try:
            self._log_writer.writerow([
                float(t_now),
                float(self.dx), float(self.dy),      # 목표 dx,dy
                float(dx_dwa), float(dy_dwa),        # 선택된 bx,by
                float(vx_cmd), float(wz_cmd),        # 퍼블리시한 vx, vyaw
                float(self.kv), float(self.kyaw),    # 게인
                "none"
            ])
            self._log_fp.flush()
        except Exception:
            pass

        # 디버그 (rate limit)
        if (t_now - self._last_log_t) > 0.3:
            self._last_log_t = t_now
            self.get_logger().info(
                f"cmd vx={cmd.linear.x:.2f} m/s, vyaw={cmd.angular.z:.2f} rad/s | "
                f"best({bx:.2f},{by:.2f}) θ={math.degrees(theta):.1f}° d={bd:.2f}"
            )

    def destroy_node(self):
        try:
            if hasattr(self, "_log_fp") and self._log_fp:
                self._log_fp.close()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DWACommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
