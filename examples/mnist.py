import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from aironsuit.suit import AIronSuit

# Example Set-Up #

num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 15

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Create model
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")])
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Training
aironsuit = AIronSuit(model=model)
aironsuit.train(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val)

# Evaluate
score = aironsuit.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
