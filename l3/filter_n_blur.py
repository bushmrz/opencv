import numpy as np
import cv2

PATH = "D:\\opencv\\opencv\\cat.jpg"


def gauss(x, y, omega, a, b):
    omegaIn2 = 2 * omega ** 2
    m1 = 1 / (np.pi * omegaIn2)
    m2 = np.exp(-((x - a) ** 2 + (y - b) ** 2) / omegaIn2)
    return m1 * m2


def getPicture():
    img = cv2.imread(PATH)
    return img


def getBlurPicture(size, deviation, img):
    kernel = np.ones((size, size))
    a = b = (size + 1) // 2

    for i in range(size):
        for j in range(size):
            kernel[i, j] = gauss(i, j, deviation, a, b)

    sum = 0
    for i in range(size):
        for j in range(size):
            sum += kernel[i, j]
    for i in range(size):
        for j in range(size):
            kernel[i, j] /= sum

    blur = img.copy()
    sx = size // 2
    sy = size // 2
    for i in range(sx, blur.shape[0] - sx):
        for j in range(sy, blur.shape[1] - sy):
            value = 0
            for k in range(-(size // 2), size // 2 + 1):
                for l in range(-(size // 2), size // 2 + 1):
                    value += img[i + k, j + l] * kernel[(size // 2) + k, (size // 2) + l]
            blur[i, j] = value

    return blur


deviation = 5
size = 5

out1 = getBlurPicture(img=getPicture(), deviation=deviation, size=size)
out2 = cv2.GaussianBlur(getPicture(), (5, 5), 5)
cv2.imshow("original", getPicture())
cv2.imshow("custom", out1)
cv2.imshow("gaussan cv2 blur", out2)

cv2.waitKey(0)
cv2.destroyAllWindows()

