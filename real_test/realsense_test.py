#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pyrealsense2 as rs
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class RealSenseRGBPublisher(Node):
    def __init__(self):
        super().__init__('realsense_rgb_pub')

        # ROS2 publisher
        self.pub = self.create_publisher(Image, '/camera/camera/rgb', 10)
        self.bridge = CvBridge()

        # RealSense pipeline 초기화
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)
        self.get_logger().info("✅ RealSense RGB publisher started.")

    def loop(self):
        try:
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # numpy array 변환
                color_image = np.asanyarray(color_frame.get_data())

                # OpenCV 창 띄워서 확인 (원하면 끄세요)
                cv2.imshow("RealSense RGB", color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # ROS2 Image 메시지 변환 후 publish
                msg = self.bridge.cv2_to_imgmsg(color_image, "bgr8")
                self.pub.publish(msg)
        except KeyboardInterrupt:
            pass
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            self.get_logger().info("🛑 RealSense stopped.")


def main():
    rclpy.init()
    node = RealSenseRGBPublisher()
    node.loop()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
