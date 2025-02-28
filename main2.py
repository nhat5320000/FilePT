import cv2
from ultralytics import YOLO

# Load YOLO model
eye_detect = YOLO("training/eyes-detect/yolov11n/best.engine", task='detect')

# Kiểm tra xem model có thông tin kích thước đầu vào không
print("🛠 Model details:", eye_detect)

# Định nghĩa nhãn
LABELS = ["OK", "NG", "NG1", "NG2", "NG3NG3"]

# Mở camera
cap = cv2.VideoCapture(0)
if not cap or not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Không thể đọc frame!")
            break

        # Lấy kích thước input phù hợp nếu model hỗ trợ
        try:
            input_size = eye_detect.overrides.get('imgsz', (640, 640))  # Mặc định dùng 640x640 nếu không có thông tin
        except AttributeError:
            input_size = (640, 320)  # Nếu lỗi, dùng 640x320

        # Resize ảnh theo model yêu cầu
        frame_resized = cv2.resize(frame, input_size)

        # Chạy model YOLO với kích thước ảnh phù hợp
        results = eye_detect(frame_resized, conf=0.7, imgsz=input_size)[0]

        # Duyệt qua từng detection
        for box in results.boxes:
            class_id = int(box.cls)
            if 0 <= class_id < len(LABELS):
                print(f"🔹 Phát hiện: {LABELS[class_id]}")

except KeyboardInterrupt:
    print("\n🛑 Nhận tín hiệu Ctrl+C, thoát chương trình...")

finally:
    cap.release()
    print("📢 Đã giải phóng camera.")
