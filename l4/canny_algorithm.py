import cv2
import numpy as np

PATH = "compcat.jpg"


def open_wb_blur_img():
    img = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("WB IMG", img)
    blur_img = cv2.GaussianBlur(img, (3, 3), 3)
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

    non_max_suppressed = np.zeros_like(edges)

    # Если текущий пиксель является локальным максимумом, он сохраняется в матрице, иначе остается пустым
    for i in range(1, edges.shape[0] - 1):
        for j in range(1, edges.shape[1] - 1):
            current_angle = angle[i, j]

            if (0 <= current_angle < np.pi / 4) or (7 * np.pi / 4 <= current_angle <= 2 * np.pi):
                prev_pixel = magnitude[i, j - 1]
                next_pixel = magnitude[i, j + 1]
            elif (np.pi / 4 <= current_angle < 3 * np.pi / 4) or (5 * np.pi / 4 <= current_angle < 7 * np.pi / 4):
                prev_pixel = magnitude[i - 1, j]
                next_pixel = magnitude[i + 1, j]
            elif (3 * np.pi / 4 <= current_angle < 5 * np.pi / 4) or (3 * np.pi / 4 <= current_angle < 5 * np.pi / 4):
                prev_pixel = magnitude[i - 1, j - 1]
                next_pixel = magnitude[i + 1, j + 1]
            elif (5 * np.pi / 4 <= current_angle < 7 * np.pi / 4) or (np.pi / 4 <= current_angle < 3 * np.pi / 4):
                prev_pixel = magnitude[i - 1, j + 1]
                next_pixel = magnitude[i + 1, j - 1]

            if magnitude[i, j] >= prev_pixel and magnitude[i, j] >= next_pixel:
                non_max_suppressed[i, j] = magnitude[i, j]

    cv2.imshow("NON MAX", non_max_suppressed)

    threshold_low = 0.1 * np.max(non_max_suppressed)  # Нижний порог (20% от максимума)
    threshold_high = 0.3 * np.max(non_max_suppressed)  # Верхний порог (60% от максимума)

    strong_edges = (non_max_suppressed > threshold_high)
    weak_edges = (non_max_suppressed >= threshold_low) & (non_max_suppressed <= threshold_high)

    H, W = strong_edges.shape

    # Проход по пикселям и установка значения в зависимости от сильного и слабого ребра
    for i in range(H):
        for j in range(W):
            if strong_edges[i, j]:
                non_max_suppressed[i, j] = 255
            elif weak_edges[i, j]:
                non_max_suppressed[i, j] = 50  # Произвольное значение для слабых ребер

    cv2.imshow("DOUBLE FILTER", non_max_suppressed)

    cv2.imshow("LENGTH MATRIX", magnitude)
    cv2.imshow("ANGLE MATRIX", angle)
    cv2.waitKey(0)


open_wb_blur_img()
