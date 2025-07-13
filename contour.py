import cv2
import numpy as np

# Đọc ảnh gốc
img = cv2.imread(r'C:\Users\Divu\Desktop\test\estimate_shrmip\shrimp4.jpg')
original = img.copy()

# Scale factor for physical unit conversion (adjust based on image calibration)
PIXELS_PER_CM = 10  # Example: 10 pixels = 1 cm

# Bước 1: Tách nền xanh -> lấy vùng tôm
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([140, 255, 255])
mask = cv2.inRange(hsv, lower_blue, upper_blue)
mask_inv = cv2.bitwise_not(mask)
cv2.imshow("Foreground", mask_inv)

# Bước 2: Morphology để làm mượt mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask_clean = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

# Bước 3: Fill các lỗ nhỏ bên trong vật thể 
final_mask = mask_clean.copy()
max_hole_area = 500

contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
if hierarchy is not None:
    for i in range(len(contours)):
        # Nếu là lỗ (có cha) và diện tích nhỏ
        if hierarchy[0][i][3] != -1:
            area = cv2.contourArea(contours[i])
            if area < max_hole_area:
                cv2.drawContours(final_mask, contours, i, 255, cv2.FILLED)

min_area = 3000
cv2.imshow("Final Mask (after filling small holes)", final_mask)

# Bước 4: Erosion nhẹ hơn
kernel = np.ones((3,3), np.uint8)
eroded = cv2.erode(final_mask, kernel, iterations=1)
cv2.imshow("Eroded", eroded)

# Bước 5: Distance transform
dist_transform = cv2.distanceTransform(eroded, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.3*dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)
cv2.imshow("Sure Foreground", sure_fg)

unknown = cv2.subtract(final_mask, sure_fg)

# Bước 6: Marker labelling
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# Watershed
markers = cv2.watershed(original, markers)

# Hiển thị marker màu để dễ debug
marker_viz = np.zeros_like(original, dtype=np.uint8)
unique_markers = np.unique(markers)
colors = {}
for marker in unique_markers:
    if marker == -1:
        colors[marker] = (0,0,255)  # biên màu đỏ
    else:
        colors[marker] = np.random.randint(0,255,3).tolist()

for marker in unique_markers:
    marker_mask = (markers == marker).astype(np.uint8) * 255
    marker_viz[marker_mask==255] = colors[marker]

cv2.imshow("Markers visualization", marker_viz)

# Bước 7: Vẽ contour, bounding box, centroid và ước lượng kích thước
output = original.copy()
shrimp_count = 0
size_estimates = []

for marker_id in range(2, np.max(markers)+1):
    mask = np.uint8(markers == marker_id)
    mask_full = cv2.bitwise_and(final_mask, final_mask, mask=mask)
    cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area > min_area:
            shrimp_count += 1
            cv2.drawContours(output, [cnt], -1, (0,255,0), 2)
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(output, (x,y), (x+w, y+h), (255,0,0), 2)
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                cv2.circle(output, (cx,cy), 5, (0,0,255), -1)
                cv2.putText(output, f"{shrimp_count}", (cx-10, cy-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            # Size estimation
            # 1. Area in square centimeters
            area_cm2 = area / (PIXELS_PER_CM ** 2)
            
            # 2. Bounding rectangle dimensions in centimeters
            width_cm = w / PIXELS_PER_CM
            height_cm = h / PIXELS_PER_CM
            
            # 3. Fitted ellipse (major and minor axes)
            major_axis_cm = minor_axis_cm = 0
            if len(cnt) >= 5:  # Need at least 5 points to fit an ellipse
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                major_axis_cm = max(axes) / PIXELS_PER_CM
                minor_axis_cm = min(axes) / PIXELS_PER_CM
                # Draw ellipse for visualization
                cv2.ellipse(output, ellipse, (255, 255, 0), 1)
            
            # Store size estimates
            size_estimates.append({
                "shrimp": shrimp_count,
                "area_cm2": area_cm2,
                "width_cm": width_cm,
                "height_cm": height_cm,
                "major_axis_cm": major_axis_cm,
                "minor_axis_cm": minor_axis_cm,
                "centroid": (cx, cy)
            })

# Bước 8: In kết quả
print(f"\n✅ Số lượng tôm đếm được: {shrimp_count}")
print("📍 Tọa độ centroid và kích thước:")
for size in size_estimates:
    print(f"  - Tôm {size['shrimp']}: centroid = {size['centroid']}, "
          f"area = {size['area_cm2']:.2f} cm², "
          f"width = {size['width_cm']:.2f} cm, height = {size['height_cm']:.2f} cm, "
          f"major_axis = {size['major_axis_cm']:.2f} cm, minor_axis = {size['minor_axis_cm']:.2f} cm")

cv2.imshow("Final Result", output)
cv2.waitKey(0)
cv2.destroyAllWindows()