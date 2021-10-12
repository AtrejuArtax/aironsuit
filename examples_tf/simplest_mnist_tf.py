# Databricks notebook source
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.net_constructors import customized_layer

# COMMAND ----------

# Example Set-Up #

project_name = 'simplest_mnist'
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 10

# COMMAND ----------

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# COMMAND ----------

# Create model
inputs = Input(shape=input_shape)
outputs = customized_layer(x=inputs, input_shape=input_shape, units=10, activation='softmax', filters=5,
                           kernel_size=15)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model=model)
aironsuit.summary()

# COMMAND ----------

# Training
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train)

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

# Save Model
aironsuit.save_model(os.path.join(os.path.expanduser("~"), project_name + '_model'))
