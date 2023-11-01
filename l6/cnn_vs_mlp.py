from keras.models import load_model
from keras.utils import to_categorical
from keras.datasets import mnist
import time
import numpy as np

# Загрузка данных MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Преобразование и нормализация данных
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32') / 255
X_test1 = X_test.reshape((X_test.shape[0], 28, 28, 1)).astype('float32') / 255
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#загрузка моделей
model_mlp = load_model(f'my_model')
model_cnn = load_model(f'cnn_model/cnn_model22.keras')

# Оцениваем производительность на тестовых данных
_,test_accuracy_mlp = model_mlp.evaluate(X_test, y_test)
_,test_accuracy_cnn = model_cnn.evaluate(X_test1, y_test)

start_time = time.time()  # Засекаем начальное время обучения
# Предсказания модели на тестовых данных
predictions_mlp = model_mlp.predict(X_test)
end_time = time.time()  # Засекаем конечное время обучения
work_time_mlp=end_time-start_time

start_time = time.time()  # Засекаем начальное время обучения
# Предсказания модели на тестовых данных
predictions_cnn = model_cnn.predict(X_test1)
end_time = time.time()  # Засекаем конечное время обучения
work_time_cnn=end_time-start_time

print(f"Модель_mlp: Процент корректной работы на тестовых данных: {round(test_accuracy_mlp*100,2)}%, "
          f" Скорость работы сети:{round(work_time_mlp,2)}")

print(f"Модель_cnn: Процент корректной работы на тестовых данных: {round(test_accuracy_cnn*100,2)}%, "
           f" Скорость работы сети:{round(work_time_cnn,2)}")
