import cv2

# Открываем видеопоток с камеры
cap = cv2.VideoCapture(cv2.CAP_DSHOW)

while True:
    # Читаем кадр с камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Отображаем кадр в окне с названием "Camera"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow("Camera", gray)

    # Если нажата клавиша "q", выходим из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()