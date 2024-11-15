import tensorflow as tf
import time
import os

# Проверка доступности GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("Using CPU for training.")

# Загрузка данных CIFAR-10
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Создание модели для CIFAR-10
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 классов для CIFAR-10
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Замер времени начала обучения
start_time = time.time()

# Обучение модели
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Замер времени окончания обучения
end_time = time.time()
print("Training time:", end_time - start_time, "seconds")

# Проверка использования GPU после обучения
if tf.config.list_physical_devices('GPU'):
    os.system("nvidia-smi")
