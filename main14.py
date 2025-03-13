import cv2
import numpy as np
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
    0: (0, 255, 0),   # Xanh lá
    1: (255, 0, 0),   # Xanh dương
    2: (0, 0, 255),   # Đỏ
    3: (255, 255, 0), # Ngà vàng
    4: (0, 255, 255)  # Lục lam
}

# Biến điều khiển
cap = None
detecting = False
show_window = False

# Hàm mở camera
def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        cap = None

# Hàm đóng camera
def close_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None

# Tạo cửa sổ điều khiển luôn hiển thị
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 300, 100)
cv2.imshow('Control Panel', np.zeros((100, 300, 3), dtype=np.uint8))

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("🛑 Thoát chương trình...")
        break
    elif key == ord('e'):
        if not detecting:
            open_camera()
            if cap is not None:
                detecting = True
                print("🔧 Bắt đầu phát hiện đối tượng!")
    elif key == ord('r'):
        if detecting:
            detecting = False
            close_camera()
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

        # Resize frame về kích thước 640x640 (để phù hợp với mô hình)
        frame_resized = cv2.resize(frame, (640, 640))
        results = object_detect(frame_resized, conf=0.11, imgsz=640)
        detections = results[0].boxes

        # Vẽ bounding box, label và giá trị confidence
        for box, cls, conf in zip(detections.xyxy.int().tolist(), detections.cls.tolist(), detections.conf.tolist()):
            x1, y1, x2, y2 = box
            color = color_map.get(int(cls), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if show_window:
            cv2.imshow("Object Detection", frame_resized)
            # Tự động tắt khi cửa sổ bị đóng (nhấn nút X)
            if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                show_window = False
    else:
        cv2.waitKey(100)  # Giảm tải CPU khi không phát hiện

# Giải phóng tài nguyên
close_camera()
cv2.destroyAllWindows()
