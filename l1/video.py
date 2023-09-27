import cv2

PATH = "bobmcat.mp4"

vid = cv2.VideoCapture(PATH)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter("bobmcat_slow.avi", fourcc, 10.0, (554, 360))
while vid.isOpened():
    ret, frame = vid.read()
    if not ret:
        print("Не могу прочитать кадр :(")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Cat bomb', cv2.resize(gray, (600, 800)))

    out.write(frame)
    if cv2.waitKey(1) == ord('q'):
        break

vid.release()
out.release()
cv2.destroyAllWindows()

