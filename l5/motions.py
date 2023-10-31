import cv2
import numpy as np

def task(kernel_size, standard_deviation, delta_tresh, min_area, name):
    video = cv2.VideoCapture('main.mov', cv2.CAP_ANY)

    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(
        img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        r'res' + str(name) + '.avi', fourcc, 25, (w, h))

    while True:
        # готовим новый кадр
        old_img = img.copy()
        is_ok, frame = video.read()
        if not is_ok:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(
            img, (kernel_size, kernel_size), standard_deviation)

        # вычисляем разницу, её бинаризируем и находим контуры
        diff = cv2.absdiff(img, old_img)
        # получаем изображение после применения пороговой бинаризации [0] и пороговое значение [1]
        thresh = cv2.threshold(diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        (contors, hierarchy) = cv2.findContours(thresh,
                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # если на кадре есть хотя бы один контур, чья площадь достаточно большая - записываем кадр
        for contr in contors:
            area = cv2.contourArea(contr)
            if area < min_area:
                continue
            video_writer.write(frame)
    video_writer.release()


kernel_size = 3
standard_deviation = 50
delta_tresh = 60 # пороговое значение
min_area = 20
task(kernel_size, standard_deviation, delta_tresh, min_area, "ver1")

# kernel_size = 11
# standard_deviation = 70
# delta_tresh = 60
# min_area = 20
# task(kernel_size, standard_deviation, delta_tresh, min_area, "ver2")
#
# kernel_size = 3
# standard_deviation = 50
# delta_tresh = 60
# min_area = 10
# task(kernel_size, standard_deviation, delta_tresh, min_area, "ver3")