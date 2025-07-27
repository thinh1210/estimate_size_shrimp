import cv2
import numpy as np
import os

def extract_shrimp_contour(img, debug=False):
    # Chuyển sang không gian màu HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Tách nền trắng nhạt
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_background = cv2.inRange(hsv, lower_white, upper_white)
    mask_shrimp = cv2.bitwise_not(mask_background)

    # Morphology làm mượt mask
    kernel = np.ones((3, 3), np.uint8)
    mask_shrimp = cv2.morphologyEx(mask_shrimp, cv2.MORPH_OPEN, kernel)
    mask_shrimp = cv2.morphologyEx(mask_shrimp, cv2.MORPH_CLOSE, kernel)

    # Làm rõ biên bằng ảnh xám
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask_shrimp = cv2.bitwise_and(mask_shrimp, mask_shrimp, mask=thresh)

    # Loại bỏ râu tôm bằng morphology opening
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    body_mask = cv2.morphologyEx(mask_shrimp, cv2.MORPH_OPEN, big_kernel, iterations=2)
    body_mask = cv2.dilate(body_mask, big_kernel, iterations=1)

    # Tìm contour
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(img)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)

    return contour_img

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # Lặp qua tất cả ảnh trong thư mục đầu vào
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            img = cv2.imread(input_path)
            if img is None:
                print(f"Không đọc được ảnh: {input_path}")
                continue

            print(f"Đang xử lý: {filename}")
            contour_img = extract_shrimp_contour(img)
            cv2.imwrite(output_path, contour_img)

    print("✅ Xử lý xong tất cả ảnh.")

# Ví dụ sử dụng
input_folder = r"D:\test\estimate_shrmip\DB3"
output_folder = r"D:\test\estimate_shrmip\contour_shrimp"
process_folder(input_folder, output_folder)
