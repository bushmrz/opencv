from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import cv2


# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование и нормализация данных
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Создание модели многослойного персептрона
# Данные будут передаваться от входного слоя к выходному слою последовательно
model = Sequential([
    # слой выполняет линейные преобразования данных и активацию ReLU
    Dense(256, activation='relu', input_shape=(28 * 28,)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

print(model.summary())  # вывод структуры НС в консоль

model.compile(
    # оптимизатор adam учитывает как скорость изменения параметров модели, так и прошлые изменения
    optimizer='adam',

    #функция потерь, которая используется для оценки ошибки между предсказанными значениями и истинными метками
    loss='categorical_crossentropy',

    #accuracy измеряет долю правильно классифицированных примеров
    metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=255, epochs=20, validation_split=0.25)
# model.fit(X_test, y_test, batch_size=64, epochs=5, validation_split=0.25)

# Оценка модели на тестовой выборке
_, test_accuracy = model.evaluate(X_test, y_test)
print('Точность на тестовом наборе данных:', test_accuracy)



def check_model(PATH: str):
    # Загрузка изображения для распознавания
    img_path = PATH  # укажите корректный путь к изображению
    img = image.load_img(img_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = img_array.reshape((1, 28 * 28)).astype('float32') / 255

    # Распознавание изображения
    prediction = model.predict(img_array)
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


if check_model('lab6.jpg') == "6":
    while check_model('t1.jpg') != "2":
        model.fit(X_train, y_train, epochs=3)
        check_model('t1.jpg')
    # Сохранение модели
    model.save("my_model_1")

