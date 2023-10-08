import cv2
import numpy as np

PATH = "compcat.jpg"


def open_wb_blur_img():
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("WB IMG", img)
    blur_img = cv2.GaussianBlur(img, (5, 5), 5)
    cv2.imshow("BLUR IMG", blur_img)

    # Алгоритм Канни для получения изображения границ, параметры 2 и 3 - пороговые значения
    edges = cv2.Canny(blur_img, 100, 200)
    cv2.imshow("EDGES IMG", edges)

    # Фильтры Sobel используются для вычисления градиента в направлениях x и y
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Вычисление матрицы длин (модуля) и матрицы угловых значений
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    angle = np.arctan2(gradient_y, gradient_x)

    cv2.imshow("LENGTH MATRIX", magnitude)
    cv2.imshow("ANGLE MATRIX", angle)
    cv2.waitKey(0)


open_wb_blur_img()
