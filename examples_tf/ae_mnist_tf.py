# Databricks notebook source
import numpy as np
from hyperopt import Trials
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from aironsuit.design.utils import choice_hp
from airontools.preprocessing import train_val_split
from airontools.constructors.models.unsupervised import ImageAE
from airontools.tools import path_management
HOME = os.path.expanduser("~")
OS_SEP = os.path.sep

# COMMAND ----------

# Example Set-Up #

model_name = 'AE_NN'
working_path = os.path.join(HOME, 'airon', model_name) + OS_SEP
num_classes = 10
batch_size = 128
epochs = 30
patience = 3
max_evals = 25
max_n_samples = None
precision = 'float32'

# COMMAND ----------

# Make/remove paths
path_management(working_path, modes=['rm', 'make'])

# COMMAND ----------

# Load and preprocess data
(train_dataset, target_dataset), _ = mnist.load_data()
if max_n_samples is not None:
    train_dataset = train_dataset[-max_n_samples:, ...]
    target_dataset = target_dataset[-max_n_samples:, ...]
train_dataset = np.expand_dims(train_dataset, -1).astype(precision) / 255

# Split data per parallel model
x_train, x_val, _, meta_val, _ = train_val_split(input_data=train_dataset, meta_data=target_dataset)

# COMMAND ----------

# AE Model constructor


def ae_model_constructor(latent_dim):

    # Create AE model and compile it
    ae = ImageAE(latent_dim)
    ae.compile(optimizer=Adam())

    return ae

# COMMAND ----------


# Training specs
train_specs = {'batch_size': batch_size}

# Hyper-parameter space
hyperparam_space = {'latent_dim': choice_hp('latent_dim', [int(val) for val in np.arange(3, 6)])}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(
    model_constructor=ae_model_constructor,
    force_subclass_weights_saver=True,
    force_subclass_weights_loader=True,
    results_path=working_path
)

# COMMAND ----------

# Automatic Model Design
print('\n')
print('Automatic Model Design \n')
aironsuit.design(
    x_train=x_train,
    x_val=x_val,
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
del x_train

# COMMAND ----------

# Get latent insights
aironsuit.visualize_representations(
    x_val,
    metadata=meta_val,
    hidden_layer_name='z',
)
