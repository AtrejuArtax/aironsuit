# Databricks notebook source
import numpy as np
from hyperopt.hp import choice
from hyperopt import Trials
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.preprocessing import train_val_split
from airontools.constructors.models.unsupervised import ImageVAE

# COMMAND ----------

# Example Set-Up #

model_name = 'VAE_NN'
num_classes = 10
batch_size = 128
epochs = 30
patience = 3
max_evals = 3
max_n_samples = None
precision = 'float32'

# COMMAND ----------

# Load and preprocess data
(train_dataset, _), _ = mnist.load_data()
if max_n_samples is not None:
    train_dataset = train_dataset[-max_n_samples:, ...]
train_dataset = np.expand_dims(train_dataset, -1).astype(precision) / 255

# Split data per parallel model
x_train, x_val, train_val_inds = train_val_split(input_data=train_dataset)

# COMMAND ----------

# VAE Model constructor


def vae_model_constructor(latent_dim):

    # Create VAE model and compile it
    vae = ImageVAE(latent_dim)
    vae.compile(optimizer=Adam())

    return vae

# COMMAND ----------


# Training specs
train_specs = {'batch_size': batch_size}

# Hyper-parameter space
hyperparam_space = {'latent_dim': choice('latent_dim', np.arange(3, 6))}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(
    model_constructor=vae_model_constructor,
    force_subclass_weights_saver=True,
    force_subclass_weights_loader=True
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
del x_train, x_val
