import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'D:\Tesseract-OCR\tesseract.exe'

def read_text_from_image(image_path):
    image = cv2.imread(image_path)

    contrasted_img = cv2.addWeighted(image, 1.5, image, -0.9, 0)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    kernel = np.array([[0, -1, 0], [-1, 9, -1], [0, -1, 0]])  # lam sac net anh
    # kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])  # lam min anh
    filt = cv2.GaussianBlur(src=contrasted_img, ksize=(3, 3), sigmaX=5, sigmaY=5)
    gray = cv2.cvtColor(filt, cv2.COLOR_BGR2GRAY)
    dilation = cv2.dilate(gray, kernel, iterations=0)
    for threshold in range(0, 255, 2):
        _, thresh = cv2.threshold(dilation, threshold, 255, cv2.THRESH_BINARY| cv2.THRESH_OTSU)
        thresh = cv2.resize(thresh, None, fx=1, fy=1)
        cv2.imshow("thresh", thresh)
        print("threshold",threshold)
        cv2.waitKey(0)
        text = pytesseract.image_to_string(thresh)
        return text
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # _, threshold_image = cv2.threshold(gray_image,111, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow("thresh", threshold_image)
    # cv2.waitKey(0)
    # text = pytesseract.image_to_string(threshold_image)
    # return text

image_path = 'screenshot_thresh7.png'

text = read_text_from_image(image_path)
print(text)
# print(len(text))
# if len(text) == 7:
#     print(text)
# else:
#     print("ERROR")