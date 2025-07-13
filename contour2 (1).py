import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

def fill_small_holes(mask, max_hole_area=500):
    """Lấp các lỗ nhỏ bên trong vật thể"""
    inv_mask = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_mask, connectivity=8)
    for i in range(1, num_labels):  # Bỏ background
        if stats[i, cv2.CC_STAT_AREA] < max_hole_area:
            mask[labels == i] = 255
    return mask

def smart_shrimp_counter(image_path, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh.")
        return

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    original = image.copy()

    # === 1. Tách nền xanh ===
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([130, 255, 255])
    background_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    foreground_mask = cv2.bitwise_not(background_mask)

    if debug: cv2.imshow("1. Foreground mask", foreground_mask)

    # === 2a. Lấp các lỗ nhỏ bên trong vật thể ===
    foreground_mask_filled = fill_small_holes(foreground_mask.copy(), max_hole_area=500)
    if debug: cv2.imshow("2a. foreground_mask + Fill holes", foreground_mask_filled)

    # === 2b. Làm sạch (OPEN + CLOSE) ===
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(foreground_mask_filled, cv2.MORPH_OPEN, kernel, iterations=4)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=4)
    if debug: cv2.imshow("2b. Clean mask", clean)
   

    # === 3. Erode để tách vật thể dính nhau ===
    eroded = cv2.erode(clean, kernel, iterations=14)
    if debug: cv2.imshow("3. Eroded mask", eroded)

    # === 4. Distance Transform và tìm local maxima ===
    blurred = cv2.GaussianBlur(eroded, (15, 15), 3)
    D = cv2.distanceTransform(blurred, cv2.DIST_L2, 5)
    local_max = peak_local_max(D, min_distance=25, labels=eroded, footprint=np.ones((20, 20)))

    mask_peaks = np.zeros_like(D, dtype=bool)
    mask_peaks[tuple(local_max.T)] = True
    markers, _ = ndimage.label(mask_peaks)

    # === 5. Áp dụng watershed ===
    labels = watershed(-D, markers, mask=eroded)

    # === 6. Mask kết quả từ watershed ===
    final_mask = np.zeros_like(clean, dtype=np.uint8)
    final_mask[labels > 0] = 255
    if debug: cv2.imshow("6. Raw mask after watershed", final_mask)

    # === 7. Dãn nở để khôi phục hình dáng + đóng khe nhỏ ===
    final_mask = cv2.dilate(final_mask, kernel, iterations=14)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    if debug: cv2.imshow("7. Final mask (Dilated + Closed)", final_mask)

    # === 8. Tìm contour, centroid và đếm ===
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    centroids = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            count += 1
            cv2.drawContours(original, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(original, (x, y), (x + w, y + h), (255, 0, 0), 1)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                cv2.circle(original, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(original, str(count), (cx - 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if debug:
                    print(f"Tôm {count}: centroid = ({cx}, {cy}), area = {area:.1f}")

    # === 9. Kết quả ===
    print(f"\n✅ Tổng số tôm đếm được: {count}")
    print("📍 Tọa độ centroid:")
    for i, (cx, cy) in enumerate(centroids, 1):
        print(f"  - Tôm {i}: ({cx}, {cy})")

    # === 10. Hiển thị ===
    cv2.imshow("9. Counting result", original)
    cv2.imshow("10. Final mask", final_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    smart_shrimp_counter(r"C:\Users\Divu\Desktop\test\estimate_shrmip\shrimp1.jpg", debug=True)
