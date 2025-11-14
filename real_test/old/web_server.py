from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import asyncio, json, math, threading, time, cv2, numpy as np
from pathlib import Path
from cv_bridge import CvBridge
import redis
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid

# 내부 모듈
from nav_utils import *
from planner_server_mac import get_gps_latlon, get_go2_xy_yawdeg, ALL_PATHS, selected_path_file

import uvicorn

# -------------------------------
# 전역 설정
# -------------------------------
app = FastAPI()
bridge = CvBridge()
r = redis.Redis(host='127.0.0.1', port=6379, db=0)
latest_color = None
latest_depth = None


# -------------------------------
# [ROS2 구독 노드]
# -------------------------------
class CameraNode(Node):
    def __init__(self):
        super().__init__('web_camera_bridge')
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.color_callback, 10)
        self.create_subscription(Image, '/camera/camera/aligned_depth_to_color/image_raw', self.depth_callback, 10)        
        self.get_logger().info("✅ Subscribed to realsense topics")

    def color_callback(self, msg):
        global latest_color
        try:
            frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            _, jpeg = cv2.imencode('.jpg', frame)
            latest_color = jpeg.tobytes()
            r.set("latest_color", latest_color)
        except Exception as e:
            self.get_logger().error(f"color_callback error: {e}")

    def depth_callback(self, msg):
        global latest_depth
        try:
            depth = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            _, jpeg = cv2.imencode('.jpg', depth_colored)
            latest_depth = jpeg.tobytes()
            r.set("latest_depth", latest_depth)
        except Exception as e:
            self.get_logger().error(f"depth_callback error: {e}")


def ros_thread():
    rclpy.init()
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
threading.Thread(target=ros_thread, daemon=True).start()


# -------------------------------
# [정적 파일 및 HTML 페이지]
# -------------------------------
app.mount("/static", StaticFiles(directory="web/static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    return open("web/planner_view.html", "r", encoding="utf-8").read()


@app.get("/map", response_class=HTMLResponse)
async def map_page():
    html_path = Path("web/map.html")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# -------------------------------
# [1] Path (waypoints)
# -------------------------------
@app.get("/selected_path", response_class=JSONResponse)
async def selected_path():
    if selected_path_file and selected_path_file in ALL_PATHS:
        return ALL_PATHS[selected_path_file]
    return {"coords": [], "file": None}


# -------------------------------
# [2] GPS 실시간 스트리밍 (SSE)
# -------------------------------
# @app.get("/gps")
# async def gps_stream():
#     async def event_gen():
#         while True:
#             try:
#                 with open("/tmp/gps_state.json", "r") as f:
#                     data = json.load(f)
#                 yield f"data: {json.dumps(data)}\n\n"
#             except Exception:
#                 pass
#             await asyncio.sleep(0.2)
#     return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/gps")
def get_gps_state():
    data = r.get("gps_state")
    if not data:
        return {"lat": None, "lon": None, "status": "no data yet"}
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return {"lat": None, "lon": None, "status": "invalid json"}

# @app.get("/gps_stream")
# def gps_stream():
#     def event_stream():
#         last = None
#         while True:
#             try:
#                 data = r.get("gps_state")
#                 if data != last and data is not None:
#                     last = data
#                     yield f"data: {data}\n\n"  # SSE 형식
#             except Exception as e:
#                 print(f"[GPS Stream] Error: {e}")
#             time.sleep(0.5)
#     return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/gps_stream")
def gps_stream():
    def event_stream():
        last = None
        while True:
            try:
                data = r.get("gps_state")
                if data and data != last:
                    # Redis에서 읽은 문자열을 JSON으로 파싱 후 다시 문자열화 (보장된 JSON)
                    parsed = json.loads(data)
                    yield f"data: {json.dumps(parsed)}\n\n"
                    last = data
            except Exception as e:
                print(f"[GPS Stream] Error: {e}")
            time.sleep(0.5)
    return StreamingResponse(event_stream(), media_type="text/event-stream")
# -------------------------------
# [3] Occupancy Grid 이미지 스트리밍
# -------------------------------
@app.get("/occupancy_grid")
async def occupancy_grid():
    png = r.get("occ_grid_latest")
    if png is None:
        print("⚠️ Redis empty")
        return Response(status_code=204)
    print("✅ Redis data found:", len(png), "bytes")
    return Response(content=png, media_type="image/png")


# -------------------------------
# [4] Realsense 카메라 스트리밍
# -------------------------------
@app.get("/color_feed")
def color_feed():
    def stream():
        while True:
            if latest_color:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_color + b"\r\n")
            time.sleep(0.05)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/depth_feed")
def depth_feed():
    def stream():
        while True:
            if latest_depth:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_depth + b"\r\n")
            time.sleep(0.05)
    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


# -------------------------------
# [5] 카메라 뷰 페이지
# -------------------------------
@app.get("/cam", response_class=HTMLResponse)
async def cam_page():
    html = """
    <html>
    <head>
        <title>Realsense Viewer</title>
        <style>
            body { background:#111; color:#eee; text-align:center; }
            h2 { margin-top: 1em; }
            img { border-radius:8px; margin:5px; }
        </style>
    </head>
    <body>
        <h2>🎥 Realsense Streams</h2>
        <img src="/color_feed" width="640">
        <img src="/depth_feed" width="640"><br>
        <a href="/" style="color:#0af;">← Back to Planner</a>
    </body>
    </html>
    """
    return HTMLResponse(html)


# -------------------------------
# 실행
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
