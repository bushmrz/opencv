import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    cross_image = np.zeros((height, width, 3), dtype=np.uint8)

    vertical_line_width = 60
    vertical_line_height = 300

    central_pixel_color = frame[height // 2, width // 2]

    color_distances = [
        np.linalg.norm(central_pixel_color - np.array([0, 0, 255])),
        np.linalg.norm(central_pixel_color - np.array([0, 255, 0])),
        np.linalg.norm(central_pixel_color - np.array([255, 0, 0]))
    ]

    closest_color_index = np.argmin(color_distances)

    points_up = [np.array([[250, 60], [400, 60], [width//2, height//2]])]
    points_down = [np.array([[250, 410], [400, 410], [width//2, height//2]])]

    points_left = [np.array([[100, 200], [100, 300], [width//2, height//2]])]
    points_right = [np.array([[550, 200], [550, 300], [width//2, height//2]])]

    cv2.fillPoly(cross_image, points_up, (0,0,255))
    cv2.fillPoly(cross_image, points_down, (0,0,255))
    cv2.fillPoly(cross_image, points_left, (0,0,255))
    cv2.fillPoly(cross_image, points_right, (0,0,255))


    if closest_color_index == 0:
         cv2.fillPoly(cross_image, points_up, (0,0,255))
         cv2.fillPoly(cross_image, points_down, (0,0,255))
         cv2.fillPoly(cross_image, points_left, (0,0,255))
         cv2.fillPoly(cross_image, points_right, (0,0,255))

    elif closest_color_index == 1:
        cv2.fillPoly(cross_image, points_up, (0, 255, 0))
        cv2.fillPoly(cross_image, points_down, (0, 255, 0))
        cv2.fillPoly(cross_image, points_left, (0, 255, 0))
        cv2.fillPoly(cross_image, points_right, (0, 255, 0))

    else:
        cv2.fillPoly(cross_image, points_up, (255, 0, 0))
        cv2.fillPoly(cross_image, points_down, (255, 0, 0))
        cv2.fillPoly(cross_image, points_left, (255, 0, 0))
        cv2.fillPoly(cross_image, points_right, (255, 0, 0))

    result_frame = cv2.addWeighted(frame, 1, cross_image, 1, 0)

    cv2.imshow("Colored Cross", result_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

