#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, math, csv
import numpy as np
import cv2
import redis
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

# Redis 연결
r = redis.Redis(host='127.0.0.1', port=6379, db=0)

class DistMapBridge(Node):
    def __init__(self):
        super().__init__('distmap_bridge_node')

        # ROS2 구독자: OccupancyGrid
        self.subscription = self.create_subscription(
            OccupancyGrid,
            '/bev/occupancy_grid',  # rosbag 토픽 이름과 동일해야 함
            self._cb_occ,
            10
        )

        # 거리맵 계산 관련 변수
        self._dist = None
        self.dist_max_m = 5.0     # 거리맵 최대 거리 [m]
        self.dist_method = 'brute' # 또는 bfs_cuda 등
        self._last_log_t = 0.0

        self.get_logger().info("✅ Subscribed to /bev/occupancy_grid")
        self.get_logger().info("✅ Redis bridge active (occ_grid_latest, dist_map_latest)")

    def _cb_occ(self, msg: OccupancyGrid):
        """ROS2 OccupancyGrid 수신 콜백"""
        H = int(msg.info.height)
        W = int(msg.info.width)
        data = np.asarray(msg.data, dtype=np.int8).reshape(H, W)
        self._occ = data
        self._info = (
            float(msg.info.resolution),
            W, H,
            float(msg.info.origin.position.x),
            float(msg.info.origin.position.y),
        )

        # ---- Redis로 PNG 전송 ----
        try:
            img = ((100 - np.clip(data, 0, 100)) * 2.55).astype(np.uint8)
            ok, png = cv2.imencode(".png", img, )#[cv2.IMWRITE_PNG_COMPRESSION, 1])
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


def main(args=None):
    rclpy.init(args=args)
    node = DistMapBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
