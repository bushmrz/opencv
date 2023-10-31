import cv2
from keras import models
from keras.preprocessing import image

# Загрузка сохраненной модели
loaded_model = models.load_model("my_model")


def check_model(PATH: str):
    # Загрузка изображения для распознавания
    img_path = PATH  # укажите корректный путь к изображению
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28 * 28)).astype('float32') / 255

    # Распознавание изображения
    prediction = loaded_model.predict(img_array)
    result = prediction.argmax()

    # Загрузка изображения и вывод результата
    img_display = cv2.imread(img_path)

    img_display_resized = cv2.resize(img_display, (256, 256))
    cv2.putText(img_display_resized, "Result: {}".format(result), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (256, 8, 0), 2)

    # Отображение измененного изображения и результатов
    cv2.imshow("result", img_display_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print('Результат распознавания:', result)
    return str(result)

check_model('lab6.jpg')
check_model('t1.jpg')


check_model('three.jpg')
check_model('t2.jpg')

