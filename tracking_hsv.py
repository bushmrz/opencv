import cv2
import numpy as np

vid = cv2.VideoCapture(0)


def erode(frame, kernel):
    m, n = frame.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(frame)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(frame[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded

def dilate(frame, kernel):
    m, n = frame.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(frame)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(frame[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated


while True:
    ret, frame = vid.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 150])  # Нижний порог оттенка, насыщенности и значения
    upper_red = np.array([10, 255, 255])  # Верхний порог оттенка, насыщенности и значения

    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)

    frame_opening = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    frame_closing = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    hsv_frame = cv2.bitwise_and(hsv_frame, hsv_frame, mask=red_mask)

    cv2.imshow("HSV", hsv_frame)
    cv2.imshow("OPEN", frame_opening)
    cv2.imshow("CLOSE", frame_closing)

    if cv2.waitKey(1) & 0xFF == 27:
        break




img = cv2.imread("cat.jpg")

def erode(image, kernel):
    m, n, _ = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded


def dilate(image, kernel):
    m, n, _ = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated


kernel = np.ones((5, 5), np.uint8)

cv2.imshow("erode", erode(img, kernel))
cv2.waitKey(0)

cv2.imshow("dilate", dilate(img, kernel))
cv2.waitKey(0)


