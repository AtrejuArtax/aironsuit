# Databricks notebook source
import os

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from aironsuit.suit import AIronSuit
from airontools.constructors.layers import layer_constructor
from airontools.tools import path_management

HOME = os.path.expanduser("~")

# COMMAND ----------

# Example Set-Up #

project_name = 'simplest_mnist'
working_path = os.path.join(HOME, 'airon', project_name)
model_name = project_name + '_NN'
num_classes = 10
batch_size = 128
epochs = 3

# COMMAND ----------

# Make/remove paths
path_management(working_path, modes=['rm', 'make'])

# COMMAND ----------

# Load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data
x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# COMMAND ----------

# Create model
input_shape = (28, 28, 1)
inputs = Input(shape=input_shape)
outputs = layer_constructor(
    x=inputs,
    input_shape=input_shape,
    units=10,
    activation='softmax',
    filters=5,
    kernel_size=15
)
model = Model(
    inputs=inputs,
    outputs=outputs)
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model=model, name=model_name)
aironsuit.summary()

# COMMAND ----------

# Training
path_management(working_path, modes=['rm', 'make'])
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
)

# COMMAND ----------

# Evaluate
score = aironsuit.model.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

# Save Model
aironsuit.model.save_weights(os.path.join(working_path, project_name + '_model'))
