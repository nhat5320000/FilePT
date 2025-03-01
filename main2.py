import cv2
from ultralytics import YOLO

# Load YOLO model
eye_detect = YOLO("training/eyes-detect/yolov11n/best.engine", task='detect')

# Định nghĩa nhãn
LABELS = ["OK", "NG", "NG1", "NG2", "NG3NG3"]

# Mở camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Không thể mở camera!")
    exit()

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Không thể đọc frame!")
            break

        # Lấy kích thước input phù hợp
        input_size = (640, 640)

        # Resize ảnh
        frame_resized = cv2.resize(frame, (input_size[1], input_size[0]))

        # Chạy YOLO
        results = eye_detect.predict(frame_resized, conf=0.4)

        # Duyệt qua từng detection và vẽ hộp giới hạn
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())  # Lấy tọa độ hộp
                class_id = int(box.cls.cpu().numpy())  # Lấy ID lớp

                if 0 <= class_id < len(LABELS):
                    label = LABELS[class_id]
                    color = (0, 255, 0)  # Màu xanh lá

                    # Vẽ hộp giới hạn lên khung hình
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, color, 2)

        # Hiển thị video
        cv2.imshow("YOLO Eye Detection", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Nhận tín hiệu Ctrl+C, thoát chương trình...")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("📢 Đã giải phóng camera.")
