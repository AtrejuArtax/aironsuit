# Databricks notebook source
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from airontools.constructors.models.unsupervised.ae import AE
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
from hyperopt import Trials

from aironsuit.design.utils import choice_hp
from aironsuit.suit import AIronSuit

HOME = os.path.expanduser("~")

# COMMAND ----------

# Example Set-Up #

model_name = "AE_NN"
working_path = os.path.join(HOME, "airon", model_name)
num_classes = 10
batch_size = 128
epochs = 3
patience = 3
max_evals = 3
max_n_samples = 1000
precision = "float32"

# COMMAND ----------

# Make/remove paths
path_management(working_path, modes=["rm", "make"])

# COMMAND ----------

# Load and preprocess data
(train_dataset, target_dataset), _ = tf.keras.datasets.mnist.load_data()
if max_n_samples is not None:
    train_dataset = train_dataset[-max_n_samples:, ...]
    target_dataset = target_dataset[-max_n_samples:, ...]
train_dataset = np.expand_dims(train_dataset, -1) / 255
n_features = np.prod(train_dataset.shape[1:])
train_dataset = train_dataset.reshape((train_dataset.shape[0], n_features))

# Split data per parallel model
x_train, x_val, _, meta_val, _ = train_val_split(
    input_data=train_dataset, meta_data=target_dataset
)

# COMMAND ----------

# AE Model constructor


def ae_model_constructor(input_shape: Tuple[int], latent_dim: int):
    # Create AE model and compile it
    ae = AE(
        input_shape=input_shape,
        latent_dim=latent_dim,
    )
    ae.compile(optimizer=tf.keras.optimizers.Adam())

    return ae


# COMMAND ----------


# Hyper-parameter space
hyperparam_space = {
    "latent_dim": choice_hp("latent_dim", [int(val) for val in np.arange(3, 6)])
}

# COMMAND ----------
model_specs = {"input_shape": x_train.shape[1:]}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(
    model_constructor=ae_model_constructor,
    results_path=working_path,
    name=model_name,
)

# COMMAND ----------

# Automatic Model Design
print("\n")
print("Automatic Model Design \n")
aironsuit.design(
    x_train=x_train,
    x_val=x_val,
    hyper_space=hyperparam_space,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    seed=0,
    patience=patience,
    metric="loss",
    model_specs=model_specs,
)
aironsuit.summary()
del x_train

# COMMAND ----------

# Get latent insights
aironsuit.visualize_representations(
    x_val,
    metadata=meta_val,
    hidden_layer_name="z",
)
