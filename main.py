import cv2

PATH = "cat.jpg"
image_bw = cv2.imread(PATH, cv2.IMREAD_GRAYSCALE)

image_color = cv2.imread(PATH, 1)
image_hsv = cv2.imread(PATH)
image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_RGB2HSV)
# cv2.imshow("IMG", image)
#
# cv2.waitKey(0)


def open_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)

    dsize = (width, height)
    output = cv2.resize(image, dsize)
    cv2.imshow("IMG", output)

    cv2.waitKey(0)

cv2.namedWindow('IMG', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('IMG', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.imshow("IMG", image_color)

cv2.waitKey(0)

cv2.destroyAllWindows()


cv2.namedWindow('IMG', cv2.WINDOW_FREERATIO)
cv2.imshow("IMG", image_bw)


cv2.waitKey(0)

cv2.destroyAllWindows()


cv2.namedWindow('IMG')
cv2.moveWindow('IMG', 100, 100)
open_image(image_bw, 0.25)
open_image(image_color, 0.5)
open_image(image_hsv, 0.7)
cv2.destroyAllWindows()

cv2.namedWindow('IMG')
cv2.moveWindow('IMG', 100, 100)
cv2.imshow("IMG", image_color)
cv2.namedWindow('IMG2')
cv2.moveWindow('IMG2', 700, 100)
cv2.imshow("IMG2", image_hsv)

cv2.waitKey(0)
cv2.destroyAllWindows()

