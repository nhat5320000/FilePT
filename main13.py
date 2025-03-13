import cv2
import numpy as np
from ultralytics import YOLO
import random

# Load mô hình YOLO cho 5 đối tượng
model_path = "training/five-objects/yolov11n/yolov11_last.engine"  # Cập nhật đường dẫn file mô hình mới
object_detect = YOLO(model_path, task='detect')

# Lấy tên các lớp từ mô hình (nếu có), nếu không thì định nghĩa thủ công
if hasattr(object_detect.model, 'names'):
    class_names = object_detect.model.names  # Ví dụ: {0: "Object1", 1: "Object2", ...}
else:
    class_names = {0: "Object1", 1: "Object2", 2: "Object3", 3: "Object4", 4: "Object5"}

# Tạo dictionary ánh xạ lớp với màu sắc cố định (bạn có thể tự định nghĩa màu hoặc sinh ngẫu nhiên)
color_map = {
    0: (0, 255, 0),   # Xanh lá
    1: (255, 0, 0),   # Xanh dương
    2: (0, 0, 255),   # Đỏ
    3: (255, 255, 0), # Ngà vàng
    4: (0, 255, 255)  # Lục lam
}

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

# Tạo cửa sổ điều khiển luôn hiển thị
cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Control Panel', 300, 100)
cv2.imshow('Control Panel', np.zeros((100, 300, 3), dtype=np.uint8))

# Biến điều khiển hiển thị
show_window = False

while True:
    success, frame = cap.read()
    if not success:
        print("❌ Không thể đọc frame!")
        break

    # Xử lý frame (resize về 320x320)
    frame_resized = cv2.resize(frame, (320, 320))
    results = object_detect(frame_resized, conf=0.11, imgsz=320)
    detections = results[0].boxes

    # Vẽ bounding box và nhãn tương ứng với đối tượng
    # Giả sử detections.xyxy chứa danh sách box và detections.cls chứa id lớp tương ứng
    for box, cls in zip(detections.xyxy.int().tolist(), detections.cls.tolist()):
        x1, y1, x2, y2 = box
        # Chọn màu dựa trên lớp; nếu không có trong color_map thì dùng màu ngẫu nhiên
        color = color_map.get(int(cls), (random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        label = class_names[int(cls)]
        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Hiển thị cửa sổ kết quả
    if show_window:
        cv2.imshow("Object Detection", frame_resized)
        # Tự động tắt khi cửa sổ bị đóng (nhấn nút X)
        if cv2.getWindowProperty("Object Detection", cv2.WND_PROP_VISIBLE) < 1:
            show_window = False
    else:
        try:
            cv2.destroyWindow("Object Detection")
        except:
            pass

    # Xử lý sự kiện phím từ cửa sổ Control Panel
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == ord('Q'):
        print("🛑 Thoát chương trình...")
        break
    elif key == ord('w') or key == ord('W'):
        show_window = not show_window
        print(f"🔧 Trạng thái hiển thị: {'BẬT' if show_window else 'TẮT'}")

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
