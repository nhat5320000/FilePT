import cv2
import numpy as np
from ultralytics import YOLO
import random
from pymodbus.client.sync import ModbusTcpClient
from pymodbus.exceptions import ConnectionException
ip_address = '192.168.4.20'
client = ModbusTcpClient(ip_address)
# Đường dẫn đến mô hình YOLO
model_path = "training/five-objects/yolov11n/yolov11_last.engine"
object_detect = YOLO(model_path, task='detect')

# Tên các lớp (cập nhật theo mô hình của bạn)
class_names = {0: "Object1", 1: "Object2", 2: "Object3", 3: "Object4", 4: "Object5"}

# Màu sắc cho các lớp
color_map = {
    0: (0, 255, 0),
    1: (255, 0, 0),
    2: (0, 0, 255),
    3: (255, 255, 0),
    4: (0, 255, 255)
}

# Biến điều khiển
detecting = True
show_window = True

# Hàm mở camera
def open_camera():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        exit()

# Mở camera ngay khi chương trình bắt đầu
open_camera()

# Tạo cửa sổ điều khiển
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 300, 100)
cv2.imshow('Control Panel', np.zeros((100, 300, 3), dtype=np.uint8))

while True:
    if detecting:
        success, frame = cap.read()
        if not success:
            print("❌ Không thể đọc frame!")
            break

        # Thực hiện phát hiện đối tượng
        frame_resized = cv2.resize(frame, (640, 640))
        results = object_detect(frame_resized, conf=0.11, imgsz=640)
        detections = results[0].boxes

        # Vẽ bounding box và nhãn
        for box, cls, conf in zip(detections.xyxy.int().tolist(), detections.cls.tolist(), detections.conf.tolist()):
            x1, y1, x2, y2 = box
            color = color_map.get(int(cls), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
            label = f"{class_names[int(cls)]}: {conf:.2f}"
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Hiển thị kết quả
        if show_window:
            cv2.imshow("Object Detection", frame_resized)
            if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
                show_window = False
        else:
            try:
                cv2.destroyWindow("Object Detection")
            except:
                pass

    M200, M201 = 0, 0  # Giá trị mặc định nếu không đọc được từ PLC

    # Kiểm tra kết nối PLC
    if client.connect():
        M200_bit = client.read_coils(8, 1)
        M201_bit = client.read_coils(9, 1)

        if not M200_bit.isError():
            M200 = M200_bit.bits[0]
        else:
            M200 = 0

        if not M201_bit.isError():
            M201 = M201_bit.bits[0]
        else:
            M201 = 0
    else:
        print("⚠️ Mất kết nối với PLC!")
    # Xử lý sự kiện phím
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("🛑 Thoát chương trình...")
        break
    elif key == ord('e') or key == ord('E') or M200:
        if not detecting:
            open_camera()
            detecting = True
            print("📸 Camera và phát hiện đối tượng BẬT")
    elif key == ord('r') or key == ord('R') or M201:
        if detecting:
            detecting = False
            cap.release()
            print("📴 Camera và phát hiện đối tượng TẮT")
    elif key == ord('w') or key == ord('W'):
        show_window = not show_window
        print(f"🔧 Trạng thái hiển thị: {'BẬT' if show_window else 'TẮT'}")

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
