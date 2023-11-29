from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import time
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
    default="haarcascade_frontalface_default.xml",
    help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
detector = cv2.CascadeClassifier(args["cascade"])

time.sleep(2.0)
PATH = "facev.mp4"
vid = cv2.VideoCapture(PATH)


def video_rec():
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    while True:
        start_time = time.time()
        ret, frame = vid.read()
        if not ret:
            print("Не могу прочитать кадр :(")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        preprocess_time = time.time() - start_time
        preprocess_times.append(preprocess_time)

        start_time = time.time()

        rects = detector.detectMultiScale(gray, scaleFactor=1.15,
            minNeighbors=8, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        inference_time = time.time() - start_time
        inference_times.append(inference_time)

        print(rects)
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        postprocess_time = time.time() - start_time
        postprocess_times.append(postprocess_time)

        speed = {
            'preprocess': sum(preprocess_times) / len(preprocess_times) if preprocess_times else 0,
            'inference': sum(inference_times) / len(inference_times),
            'postprocess': sum(postprocess_times) / len(postprocess_times) if postprocess_times else 0
        }

        print("speed:", speed)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()

    total_preprocess_time = sum(preprocess_times)
    total_inference_time = sum(inference_times)
    total_postprocess_time = sum(postprocess_times)

    print("speed:")
    print("total preprocess time:", total_preprocess_time)
    print("total inference time:", total_inference_time)
    print("total postprocess time:", total_postprocess_time)

def camera_detect():
    vs = VideoStream(src=0).start()
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector.detectMultiScale(gray, scaleFactor=1.05,
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)

        print(rects)
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()

# camera_detect()
video_rec()

