import cv2
import numpy as np
import time
from ultralytics import YOLO
import random

# Đường dẫn đến mô hình YOLO
model_path = "training/five-objects/yolov11n/yolov11_last.engine"  # Cập nhật đường dẫn file mô hình mới
object_detect = YOLO(model_path, task='detect')

# Lấy tên các lớp từ mô hình (nếu có), nếu không thì định nghĩa thủ công
if hasattr(object_detect.model, 'names'):
    class_names = object_detect.model.names
else:
    class_names = {0: "Object1", 1: "Object2", 2: "Object3", 3: "Object4", 4: "Object5"}

# Tạo dictionary ánh xạ lớp với màu sắc cố định
color_map = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255)
}

def open_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return None
    return cap

cap = None
detecting = False
show_window = False

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("🛑 Thoát chương trình...")
        break
    elif key == ord('e'):
        if not detecting:
            cap = open_camera()
            if cap is not None:
                detecting = True
                print("🔧 Bắt đầu phát hiện đối tượng!")
    elif key == ord('r'):
        if detecting:
            detecting = False
            if cap is not None:
                cap.release()
                cap = None
            if show_window:
                cv2.destroyWindow("Object Detection")
            print("🛑 Dừng phát hiện đối tượng và tắt camera.")
    elif key == ord('w'):
        if detecting:
            show_window = not show_window
            if not show_window:
                cv2.destroyWindow("Object Detection")
            print(f"🔧 Trạng thái hiển thị: {'BẬT' if show_window else 'TẮT'}")

    if detecting and cap is not None:
        success, frame = cap.read()
        if not success:
            print("❌ Không thể đọc frame!")
            break

        frame_resized = cv2.resize(frame, (640, 640))
        results = object_detect(frame_resized, conf=0.11, imgsz=640)
        detections = results[0].boxes

        for box, cls, conf in zip(detections.xyxy.int().tolist(), detections.cls.tolist(), detections.conf.tolist()):
            x1, y1, x2, y2 = box
            color = color_map.get(int(cls), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show_window:
            cv2.imshow("Object Detection", frame_resized)

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
