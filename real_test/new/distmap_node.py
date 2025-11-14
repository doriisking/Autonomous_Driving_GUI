#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid

#from distmap_def import build_dist_map_bruteforce, build_dist_map_bfs, distmap_to_occupancygrid
latest_grid_msg = None
class DistMapPublisher(Node):
    def __init__(self):
        super().__init__('distmap_publisher')

        # ① 구독: OccupancyGrid 입력 (/bev/occupancy_grid)
        self.subscriber = self.create_subscription(
            OccupancyGrid,
            '/bev/occupancy_grid',
            self.listener_callback,
            10
        )
        # ② 퍼블리셔: 거리맵 출력 (/dist_map)
        self.publisher = self.create_publisher(
            OccupancyGrid,
            '/dist_map',
            10
        )

        self.get_logger().info("✅ Subscribed to /bev/occupancy_grid")
        self.get_logger().info("✅ Publishing distance map to /dist_map")
    
    def listener_callback(self, msg):
        global latest_grid_msg
        latest_grid_msg = msg
        self.get_logger().info(f"📥 Received OccupancyGrid ({msg.info.width}x{msg.info.height})")
        
        
    def map_callback(self, msg: OccupancyGrid):
        self.get_logger().info("Received OccupancyGrid, computing distance map...")

        # 거리맵 계산 1 (C++ buildDistMap()과 동일)
        ## dist_map = build_dist_map_bruteforce(msg, max_dist=2.0)

        # 거리맵 계산 2 
       # dist_map = build_dist_map_bfs(msg,max_dist=2.0)
        # OccupancyGrid 형식으로 변환
        #dist_msg = distmap_to_occupancygrid(dist_map, msg, max_dist=2.0)

        # 퍼블리시
        #self.publisher.publish(dist_msg)
        self.get_logger().info("Published /dist_map")

def main(args=None):
    rclpy.init(args=args)
    node = DistMapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

