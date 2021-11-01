# Databricks notebook source
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.model_constructors import layer_constructor
from airontools.tools import path_management
HOME = os.path.expanduser("~")

# COMMAND ----------

# Example Set-Up #

project_name = 'simple_mnist'
working_path = os.path.join(HOME, project_name)
num_classes = 10
batch_size = 128
epochs = 20

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
outputs = layer_constructor(x=inputs, input_shape=input_shape, units=10, activation='softmax', filters=5,
                            kernel_size=15)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model=model)

# COMMAND ----------

# Training
path_management(working_path, modes=['rm', 'make'])
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
    path=working_path)
aironsuit.summary()

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

# Save Model
aironsuit.save_model(os.path.join(working_path, project_name + '_model'))
del aironsuit, model

# COMMAND ----------

# Re-Invoke AIronSuit and load model
aironsuit = AIronSuit()
aironsuit.load_model(os.path.join(working_path, project_name + '_model'))
aironsuit.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Further Training
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train)

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
