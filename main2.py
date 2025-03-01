import json
from typing import Generator

import cv2
import tensorrt as trt
import torch
import uvicorn
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from torchvision import transforms
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Global variables for models
object_detect_model = None  # YOLO-based object detection model
context = None  # TensorRT execution context
input_shape = None  # Input shape for the model
output_shape = None  # Output shape for the model
logits = None  # Tensor to hold model output

def load_trt_engine(engine_path: str):
    """
    Load a TensorRT engine from the specified file path.

    Args:
        engine_path (str): Path to the TensorRT engine file.

    Returns:
        trt.ICudaEngine: The deserialized TensorRT engine.
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
        try:
            meta_len = int.from_bytes(f.read(4), byteorder='little')
            metadata = json.loads(f.read(meta_len).decode('utf-8'))
            print('Model metadata:', metadata)
        except UnicodeDecodeError:
            f.seek(0)
        return runtime.deserialize_cuda_engine(f.read())

def generate_video(source: str, detect_conf: float) -> Generator[bytes, None, None]:
    """
    Stream video frames with object detection.

    Args:
        source (str): Video source (camera index or file path).
        detect_conf (float): Confidence threshold for object detection.

    Yields:
        bytes: Encoded video frame.
    """
    global object_detect_model

    video_source = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="Source not found or cannot be accessed.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_results = object_detect_model.predict(source=frame, stream=True, conf=detect_conf, imgsz=320, save=False)

        for detect_result in detect_results:
            boxes = detect_result.boxes.xyxy.cpu().to(dtype=torch.int).tolist()
            labels = detect_result.boxes.cls.cpu().to(dtype=torch.int).tolist()
            names = object_detect_model.names
            
            for box, label in zip(boxes, labels):
                class_name = names[label] if label < len(names) else "Unknown"
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                cv2.putText(frame, class_name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.on_event("startup")
def load_models():
    """
    Load the object detection model during application startup.
    """
    global object_detect_model

    object_detect_model = YOLO("training/object-detect/best.engine", task='detect')

@app.get("/video_feed")
def video_feed():
    """
    Serve the video feed with object detection.

    Returns:
        StreamingResponse: Video stream response.
    """
    return StreamingResponse(generate_video(source="0", detect_conf=0.7),
        media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Serve the home page.

    Returns:
        HTMLResponse: Home page content.
    """
    return HTMLResponse(content=open("src/index.html").read())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
