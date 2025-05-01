import cv2
import numpy as np
import os
from datetime import datetime

# Thư mục lưu ảnh
save_dir = "captured_images"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Các tham số cho zoom, crop và xoay
zoom_factor = 1.0  # Hệ số phóng đại (1.0 là kích thước gốc)
rotate_angle = 0  # Góc xoay, mặc định là 0 (không xoay)
flip_code = 1  # 1: Lật ảnh theo chiều ngang (hoặc theo trục dọc nếu flip_code = 0)

# Biến lưu trữ thông tin về crop
crop_start_point = None
crop_end_point = None
is_drawing = False
dragging = False  # Biến để kiểm tra nếu đang kéo góc crop

# Hàm mở camera
def open_camera():
    global cap
    gst_str = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=(int)640, height=(int)640, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("❌ Không thể kết nối CSI camera!")
        exit()

# Hàm Zoom ảnh
def zoom_image(image, factor):
    height, width = image.shape[:2]
    new_width = int(width * factor)
    new_height = int(height * factor)
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image

# Hàm Xoay ảnh
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]))
    return rotated_image

# Hàm lật ảnh
def flip_image(image, flip_code):
    return cv2.flip(image, flip_code)

# Hàm xử lý sự kiện chuột (cho việc crop ảnh)
def mouse_callback(event, x, y, flags, param):
    global crop_start_point, crop_end_point, is_drawing, dragging

    if event == cv2.EVENT_LBUTTONDOWN:
        # Bắt đầu vẽ hoặc kéo góc crop
        if crop_start_point and crop_end_point:
            # Kiểm tra nếu điểm click gần góc crop thì cho phép kéo
            if abs(x - crop_start_point[0]) < 10 and abs(y - crop_start_point[1]) < 10:
                dragging = True
                crop_start_point = (x, y)
            elif abs(x - crop_end_point[0]) < 10 and abs(y - crop_end_point[1]) < 10:
                dragging = True
                crop_end_point = (x, y)
        else:
            crop_start_point = (x, y)
            is_drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            crop_end_point = (x, y)
            temp_image = frame.copy()
            cv2.rectangle(temp_image, crop_start_point, crop_end_point, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", temp_image)
        elif dragging:
            # Di chuyển các góc khi kéo chuột
            if crop_start_point:
                crop_end_point = (x, y)
            temp_image = frame.copy()
            cv2.rectangle(temp_image, crop_start_point, crop_end_point, (0, 255, 0), 2)
            cv2.imshow("Camera Feed", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        # Kết thúc vẽ khi nhả chuột
        if not dragging:
            crop_end_point = (x, y)
        is_drawing = False
        dragging = False
        print(f"Vùng crop: {crop_start_point} -> {crop_end_point}")

# Mở camera
open_camera()  # Mở camera ngay khi khởi động

# Tạo cửa sổ điều khiển
cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Camera Feed', 960, 540)  # Đảm bảo cửa sổ có kích thước phù hợp
cv2.setMouseCallback('Camera Feed', mouse_callback)  # Gắn sự kiện chuột

while True:
    success, frame = cap.read()
    
    # Kiểm tra nếu frame không hợp lệ
    if not success or frame is None:
        print("❌ Lỗi đọc frame từ camera!")
        continue  # Bỏ qua vòng lặp này và đọc lại frame

    # Kiểm tra kích thước ảnh hợp lệ
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        print("❌ Kích thước ảnh không hợp lệ!")
        continue  # Bỏ qua vòng lặp này và đọc lại frame

    # Zoom ảnh
    frame_zoomed = zoom_image(frame, zoom_factor)

    # Xoay ảnh
    frame_rotated = rotate_image(frame_zoomed, rotate_angle)

    # Lật ảnh
    frame_flipped = flip_image(frame_rotated, flip_code)

    # Hiển thị ảnh sau khi xử lý
    if crop_start_point and crop_end_point:
        # Vẽ hình chữ nhật crop trên ảnh khi có điểm bắt đầu và kết thúc
        cv2.rectangle(frame_flipped, crop_start_point, crop_end_point, (0, 255, 0), 2)

    # Kiểm tra kích thước hợp lệ của ảnh trước khi hiển thị
    if frame_flipped.shape[0] > 0 and frame_flipped.shape[1] > 0:
        cv2.imshow("Camera Feed", frame_flipped)
    else:
        print("❌ Lỗi: Kích thước ảnh không hợp lệ!")

    # Xử lý phím điều khiển
    key = cv2.waitKey(1) & 0xFF
    if key in [ord('q'), ord('Q')]:
        print("🛑 Thoát chương trình...")
        break
    elif key in [ord('s'), ord('S')]:
        # Lưu ảnh khi nhấn phím 's'
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = os.path.join(save_dir, f"image_{timestamp}.jpg")
        
        if crop_start_point and crop_end_point:
            # Cắt ảnh theo vùng crop
            x1, y1 = crop_start_point
            x2, y2 = crop_end_point
            cropped_image = frame_flipped[y1:y2, x1:x2]
            cv2.imwrite(img_filename, cropped_image)
            print(f"📸 Đã lưu ảnh crop: {img_filename}")
        else:
            cv2.imwrite(img_filename, frame_flipped)
            print(f"📸 Đã lưu ảnh đầy đủ: {img_filename}")
    elif key == ord('+'):  # Phóng to ảnh
        zoom_factor += 0.1
        print(f"🔍 Tỉ lệ zoom: {zoom_factor}")
    elif key == ord('-'):  # Thu nhỏ ảnh
        zoom_factor -= 0.1
        print(f"🔍 Tỉ lệ zoom: {zoom_factor}")
    elif key == ord('r'):  # Xoay ảnh theo góc
        rotate_angle += 90
        print(f"🔄 Góc xoay: {rotate_angle}")
    elif key == ord('f'):  # Lật ảnh
        flip_code = 1 if flip_code != 1 else 0
        print(f"🔄 Lật ảnh: {'Chưa lật' if flip_code == 0 else 'Lật'}")
    elif key == ord('c'):  # Xóa crop khi nhấn phím 'C'
        crop_start_point = None
        crop_end_point = None
        print("❌ Đã xóa vùng crop.")

# Dọn dẹp
cap.release()
cv2.destroyAllWindows()
