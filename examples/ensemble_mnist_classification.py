# Databricks notebook source
import os
import pickle

import numpy as np
from airontools.constructors.layers import layer_constructor
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
from hyperopt import Trials
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from aironsuit.design.utils import choice_hp, uniform_hp
from aironsuit.suit import AIronSuit

HOME = os.path.expanduser("~")

# COMMAND ----------

# Example Set-Up #

project_name = "simple_mnist_classifier"
working_path = os.path.join(HOME, "airon", project_name)
model_name = project_name + "_NN"
num_classes = 10
batch_size = 32
epochs = 1
patience = 3
max_evals = 1
precision = "float32"

# COMMAND ----------

# Make/remove paths
path_management(working_path, modes=["rm", "make"])

# COMMAND ----------

# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Split data per parallel model
x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
    input_data=x_train, output_data=y_train
)

# COMMAND ----------

# Classifier Model constructor


class Ensemble(object):
    def __init__(self, **kwargs):
        # AIron NN
        inputs = Input(shape=(28, 28, 1))
        outputs = layer_constructor(
            x=inputs,
            filters=kwargs[
                "filters"
            ],  # Number of filters used for the convolutional layer
            kernel_size=(
                kwargs["kernel_size"],
                kwargs["kernel_size"],
            ),  # Kernel size used for the convolutional layer
            strides=2,  # Strides used for the convolutional layer
            sequential_axis=-1,  # Channel axis, used to define the sequence for the self-attention layer
            num_heads=kwargs[
                "num_heads"
            ],  # Self-attention heads applied after the convolutional layer
            units=10,  # Dense units applied after the self-attention layer
            activation="softmax",  # Output activation function
            advanced_reg=True,
        )
        self.airon_nn = Model(inputs=inputs, outputs=outputs)
        self.airon_nn_impact = kwargs["airon_nn_impact"]
        self.__airon_nn_compile()

        # SKLearn NN
        self.sklearn_nn = MLPClassifier()
        self.sklearn_nn_impact = 1 - self.airon_nn_impact

    def __airon_nn_compile(
        self,
    ):
        self.airon_nn.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics="accuracy"
        )

    def compile(self):
        self.__airon_nn_compile()

    def fit(self, x, y, **kwargs):
        self.airon_nn.fit(x, y, **kwargs)
        self.sklearn_nn.fit(np.reshape(x, (len(x), np.prod(x.shape[1:]))), y)

    def evaluate(self, x, y, **kwargs):
        accuracy = self.airon_nn.evaluate(x, y, **kwargs)[-1] * self.airon_nn_impact
        accuracy += (
            accuracy_score(
                y,
                self.sklearn_nn.predict(np.reshape(x, (len(x), np.prod(x.shape[1:])))),
            )
            * self.sklearn_nn_impact
        )
        return accuracy

    def save_weights(self, path):
        self.airon_nn.save_weights(os.path.join(path, "airon_nn"))
        pickle.dump(self.sklearn_nn, open(os.path.join(path, "sklearn_nn"), "wb"))

    def load_weights(self, path):
        self.airon_nn.load_weights(os.path.join(path, "airon_nn"))
        self.sklearn_nn = pickle.load(open(os.path.join(path, "sklearn_nn"), "rb"))


# COMMAND ----------

# Hyper-parameter space
hyperparam_space = {
    "filters": choice_hp("filters", [int(val) for val in np.arange(3, 30)]),
    "kernel_size": choice_hp("kernel_size", [int(val) for val in np.arange(3, 10)]),
    "num_heads": choice_hp("num_heads", [int(val) for val in np.arange(2, 10)]),
    "airon_nn_impact": uniform_hp("airon_nn_impact", 0.1, 0.9),
}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(
    model_constructor=Ensemble,
    name=model_name,
)

# COMMAND ----------

# Automatic Model Design
print("\n")
print("Automatic Model Design \n")
aironsuit.design(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    hyper_space=hyperparam_space,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    seed=0,
    patience=patience,
)

# COMMAND ----------

# Evaluate
print("Test accuracy:", aironsuit.evaluate(x_test, y_test))
