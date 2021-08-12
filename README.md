# AIronSuit

AIronSuit (Beta) is a Python library for automatic model design/selection and visualization purposes built to work with 
[tensorflow](https://github.com/tensorflow/tensorflow) (or [pytorch](https://github.com/pytorch/pytorch) in the future) 
as a backend. It aims to accelerate
the development of deep learning approaches for research/development purposes by providing components relying on cutting 
edge approaches. It is flexible and its components can be 
replaced by customized ones from the user. The user mostly focuses on defining the input and output, 
and AIronSuit takes care of its optimal mapping. 

Key features:

1. Automatic model design/selection with [hyperopt](https://github.com/hyperopt/hyperopt). 
2. Parallel computing for multiple models across multiple GPUs when using a k-fold approach.
3. Built-in model trainer that saves training progression to be visualized with 
   [TensorBoard](https://github.com/tensorflow/tensorboard).
4. Machine learning tools from [AIronTools](https://github.com/AtrejuArtax/airontools): `net_constructor`, `custom_block`, 
   `custom_layer`, preprocessing utils, etc.
5. Flexibility: the user can replace AIronSuit components by a user customized one. For instance,
    the net constructor can be easily replaced by a user customized one.
   
### Installation

`pip install aironsuit`

### Example

``` python
# Databricks notebook source
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit

# COMMAND ----------

# Example Set-Up #

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
model = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# COMMAND ----------

# Training
aironsuit = AIronSuit(model=model)
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train)
aironsuit.summary()

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```

### More Examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)