# Databricks notebook source
import numpy as np
from hyperopt import Trials
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from aironsuit.design.utils import choice_hp
from airontools.constructors.layers import layer_constructor
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
HOME = os.path.expanduser("~")
OS_SEP = os.path.sep

# COMMAND ----------

# Example Set-Up #

project_name = 'simple_mnist_classifier'
working_path = os.path.join(HOME, 'airon', project_name) + OS_SEP
model_name = project_name + '_NN'
num_classes = 10
batch_size = 32
epochs = 25
patience = 3
max_evals = 25
precision = 'float32'

# COMMAND ----------

# Make/remove paths
path_management(working_path, modes=['rm', 'make'])

# COMMAND ----------

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Split data per parallel model
x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
    input_data=x_train,
    output_data=y_train
)

# COMMAND ----------

# Classifier Model constructor

model_specs = {
    'input_shape': (28, 28, 1),
    'loss': 'categorical_crossentropy',
    'optimizer': 'adam',
    'metrics': ['accuracy']
}


def classifier_model_constructor(**kwargs):

    inputs = Input(shape=kwargs['input_shape'])
    outputs = layer_constructor(
        x=inputs,
        filters=kwargs['filters'],  # Number of filters used for the convolutional layer
        kernel_size=(kwargs['kernel_size'],
                     kwargs['kernel_size']),  # Kernel size used for the convolutional layer
        strides=2,  # Strides used for the convolutional layer
        sequential_axis=-1,  # Channel axis, used to define the sequence for the self-attention layer
        num_heads=kwargs['num_heads'],  # Self-attention heads applied after the convolutional layer
        units=10,  # Dense units applied after the self-attention layer
        activation='softmax',  # Output activation function
        advanced_reg=True
    )
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=kwargs['loss'],
        optimizer=kwargs['optimizer'],
        metrics=kwargs['metrics']
    )

    return model

# COMMAND ----------


# Training specs
train_specs = {'batch_size': batch_size}

# Hyper-parameter space
hyperparam_space = {
    'filters': choice_hp('filters', [int(val) for val in np.arange(3, 30)]),
    'kernel_size': choice_hp('kernel_size', [int(val) for val in np.arange(3, 10)]),
    'num_heads': choice_hp('num_heads', [int(val) for val in np.arange(2, 10)])
}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model_constructor=classifier_model_constructor)

# COMMAND ----------

# Automatic Model Design
print('\n')
print('Automatic Model Design \n')
aironsuit.design(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    model_specs=model_specs,
    hyper_space=hyperparam_space,
    train_specs=train_specs,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    name=model_name,
    seed=0,
    patience=patience
)
aironsuit.summary()

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# COMMAND ----------

# Save Model
aironsuit.save_model(os.path.join(working_path, project_name + '_model'))
del aironsuit

# COMMAND ----------

# Re-Invoke AIronSuit and load model
aironsuit = AIronSuit()
aironsuit.load_model(os.path.join(working_path, project_name + '_model'))
aironsuit.model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Further Training
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train
)

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
