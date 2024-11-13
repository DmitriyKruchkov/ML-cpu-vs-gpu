import tensorflow as tf
import time
import os

# Проверка доступности GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU is available!")
else:
    print("Using CPU for training.")

# Загрузка данных MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

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

if tf.config.list_physical_devices('GPU'):
    os.system("nvidia-smi")
