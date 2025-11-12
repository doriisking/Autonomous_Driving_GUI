#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge
import struct


class RealSenseCameraNode(Node):
    def __init__(self):
        super().__init__('realsense_camera_node')

        # ---------- Publishers ----------
        self.pub_color = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.pub_depth = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)
        self.pub_info  = self.create_publisher(CameraInfo, '/camera/color/camera_info', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/camera/depth_registered/points', 10)

        self.bridge = CvBridge()

        # ---------- RealSense Pipeline ----------
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # ---------- 카메라 내부 파라미터 ----------
        profile = self.pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color)
        intr = color_profile.as_video_stream_profile().get_intrinsics()

        self.cam_info = CameraInfo()
        self.cam_info.width  = intr.width
        self.cam_info.height = intr.height
        self.cam_info.k = [intr.fx, 0.0, intr.ppx,
                           0.0, intr.fy, intr.ppy,
                           0.0, 0.0, 1.0]
        self.cam_info.d = intr.coeffs
        self.cam_info.distortion_model = "plumb_bob"

        self.get_logger().info("✅ RealSense Camera Node started.")

        # ---------- Timer (30Hz) ----------
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

    def timer_callback(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # ---------- Publish Color ----------
        msg_color = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
        msg_color.header.frame_id = "camera_color_optical_frame"
        self.pub_color.publish(msg_color)

        # ---------- Publish Depth ----------
        msg_depth = self.bridge.cv2_to_imgmsg(depth_image, "16UC1")
        msg_depth.header.frame_id = "camera_depth_optical_frame"
        self.pub_depth.publish(msg_depth)

        # ---------- Publish CameraInfo ----------
        self.cam_info.header.stamp = msg_color.header.stamp
        self.cam_info.header.frame_id = "camera_color_optical_frame"
        self.pub_info.publish(self.cam_info)

        # ---------- Generate & Publish PointCloud ----------
        points = self.pointcloud_from_depth(depth_image, color_image)
        if points is not None:
            self.pub_cloud.publish(points)

    def pointcloud_from_depth(self, depth_image, color_image):
        """Create a colored PointCloud2 from aligned depth and color images."""
        h, w = depth_image.shape
        fx = self.cam_info.k[0]
        fy = self.cam_info.k[4]
        cx = self.cam_info.k[2]
        cy = self.cam_info.k[5]

        depth_scale = 0.001  # meters per unit (depends on device)
        zs = depth_image * depth_scale
        mask = zs > 0

        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        xs = (xs - cx) * zs / fx
        ys = (ys - cy) * zs / fy

        xyz = np.stack((xs, ys, zs), axis=-1)
        xyz = xyz[mask]

        rgb = color_image[mask]
        rgb = np.left_shift(rgb[:, 0], 16) + np.left_shift(rgb[:, 1], 8) + rgb[:, 2]

        # Create PointCloud2
        msg = PointCloud2()
        msg.header.frame_id = "camera_link"

        msg.height = 1
        msg.width = xyz.shape[0]
        msg.is_bigendian = False
        msg.is_dense = True

        msg.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * xyz.shape[0]

        data = np.zeros((xyz.shape[0], 4), dtype=np.float32)
        data[:, :3] = xyz
        data[:, 3] = rgb.view(np.float32)
        msg.data = data.tobytes()
        return msg


def main():
    rclpy.init()
    node = RealSenseCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
