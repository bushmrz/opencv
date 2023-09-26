import cv2
url = "http://192.168.66.147:8080"
vs = cv2.VideoCapture(url+"/video")

while True:
    ret, frame = vs.read()
    if not ret:
        continue
    resize = cv2.resize(frame, (600,600))
    cv2.imshow('Frame', resize)
    if cv2.waitKey(1) & 0xFF == 27:
        break