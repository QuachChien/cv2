import cv2
import numpy as np
import pytesseract
# Đọc ảnh từ file

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'
config = '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ</'


# image = cv2.imread('screenshot_thresh7.png')

# image_path = 'screenshot_thresh7.png'
# image = cv2.imread(image_path)

# image_path = 'screenshot_thresh7.png'
# image = cv2.imread(image_path)
# zoom_factor = 4
# screenshot_cv_zoomed = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
# # xoay ảnh
# screenshot_cv_rotated = cv2.rotate(screenshot_cv_zoomed, cv2.ROTATE_90_CLOCKWISE)
# # Làm nét ảnh
# screenshot_cv_blurred = cv2.GaussianBlur(screenshot_cv_rotated, (5, 5), 0)
# screenshot_cv_blurred = cv2.bilateralFilter(screenshot_cv_rotated, 9, 75, 75)
# huong = (3, 3)
# kernel3 = np.ones((3, 2), np.uint8)
# filt = cv2.GaussianBlur(src=screenshot_cv_zoomed, ksize=huong, sigmaX=5, sigmaY=5)
# gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
# _, thresh = cv2.threshold(gray, 55, 255, cv2.THRESH_BINARY)



# Chuyển đổi ảnh sang ảnh grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(thresh,(5, 5), 0)
_, thresholded_image = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY_INV)

kernel_dilation = np.ones((1, 1), np.uint8)
erode = cv2.erode(thresholded_image, kernel_dilation, iterations=3)

height, width = image.shape[:2]
angle = 15
rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

rotated_image = cv2.warpAffine(erode, rotation_matrix, (width, height))

text = pytesseract.image_to_string(thresholded_image)
print('text: ', text)

# cv2.imshow('thresh', rotated_image)
cv2.imshow('Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()