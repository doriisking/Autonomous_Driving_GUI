from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
import asyncio, json, random, io
import cv2, numpy as np

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
async def index():
    return open("index.html", "r", encoding="utf-8").read()

@app.get("/gps")
async def gps_stream():
    async def event_gen():
        lat, lon = 37.5665, 126.9780
        heading = 0.0
        while True:
            lat += random.uniform(-0.00002, 0.00002)
            lon += random.uniform(-0.00002, 0.00002)
            heading = (heading + random.uniform(-2, 2)) % 360
            payload = {"lat": lat, "lon": lon, "heading": heading}
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.3)
    return StreamingResponse(event_gen(), media_type="text/event-stream")

@app.get("/camera")
async def camera_stream():
    async def gen():
        boundary = b"--frame\r\n"
        while True:
            # 무작위 색상 프레임 생성
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            color = tuple(random.randint(0, 255) for _ in range(3))
            img[:] = color
            cv2.putText(img, "Test Camera", (60, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            ok, buf = cv2.imencode(".jpg", img)
            if not ok: continue
            yield boundary
            yield b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            await asyncio.sleep(0.2)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


## pip install fastapi uvicorn opencv-python-headless numpy
## uvicorn main:app --reload