# Databricks notebook source
import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from airontools.constructors.layers import layer_constructor
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
from hyperopt import Trials

from aironsuit.design.utils import choice_hp
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
(train_dataset, train_target), (
    test_dataset,
    test_target,
) = tf.keras.datasets.mnist.load_data()
train_dataset = np.expand_dims(train_dataset.astype("float32") / 255, -1)
test_dataset = np.expand_dims(test_dataset.astype("float32") / 255, -1)
n_features = np.prod(train_dataset.shape[1:])
train_target = tf.keras.utils.to_categorical(train_target, num_classes)
test_target = tf.keras.utils.to_categorical(test_target, num_classes)

# Split data per parallel model
x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
    input_data=train_dataset,
    output_data=train_target,
)

# COMMAND ----------

# Classifier Model constructor

model_specs = {
    "input_shape": train_dataset.shape[1:],
    "loss": "categorical_crossentropy",
    "optimizer": "adam",
    "metrics": ["accuracy"],
}


def classifier_model_constructor(
    input_shape: Tuple[int],
    kernel_size: int,
    filters: int,
    num_heads: int,
    loss: str,
    optimizer: str,
    metrics: List[str],
):
    inputs = tf.keras.layers.Input(shape=input_shape)
    outputs = layer_constructor(
        x=inputs,
        filters=filters,  # Number of filters used for the convolutional layer
        kernel_size=kernel_size,  # Kernel size used for the convolutional layer
        strides=2,  # Strides used for the convolutional layer
        sequential_axis=-1,  # Channel axis, used to define the sequence for the self-attention layer
        num_heads=num_heads,  # Self-attention heads applied after the convolutional layer
        units=10,  # Dense units applied after the self-attention layer
        activation="softmax",  # Output activation function
    )
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    return model


# COMMAND ----------


# Hyper-parameter space
hyperparam_space = {
    "filters": choice_hp("filters", [int(val) for val in np.arange(3, 30)]),
    "kernel_size": choice_hp("kernel_size", [int(val) for val in np.arange(3, 10)]),
    "num_heads": choice_hp("num_heads", [int(val) for val in np.arange(2, 10)]),
}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(
    model_constructor=classifier_model_constructor,
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
    model_specs=model_specs,
    hyper_space=hyperparam_space,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    seed=0,
    patience=patience,
    metric="loss",
)
aironsuit.summary()

# COMMAND ----------

# Evaluate
score = aironsuit.model.evaluate(test_dataset, test_target)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# COMMAND ----------

# Save Model
aironsuit.model.save_weights(os.path.join(working_path, model_name))
best_model_specs = model_specs.copy()
best_model_specs.update(aironsuit.load_hyper_candidates())
del aironsuit

# COMMAND ----------

# Re-Invoke AIronSuit and load model
aironsuit = AIronSuit(
    model=classifier_model_constructor(**best_model_specs),
    name=model_name,
)
aironsuit.model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
aironsuit.model.load_weights(os.path.join(working_path, model_name))

# Further Training
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
)

# COMMAND ----------

# Evaluate
score = aironsuit.model.evaluate(test_dataset, test_target)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
