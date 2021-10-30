# Databricks notebook source
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Layer, Reshape, Conv2DTranspose
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from hyperopt.hp import uniform, choice
from hyperopt import Trials
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.preprocessing import array_to_list

# COMMAND ----------

# Example Set-Up #

model_name = 'VAE_NN'
num_classes = 10
batch_size = 128
epochs = 100
patience = 3
max_evals = 3
max_n_samples = None

# COMMAND ----------

# Load and preprocess data
(train_dataset, train_targets), _ = mnist.load_data()
if max_n_samples is not None:
    train_dataset = train_dataset[-max_n_samples:, ...]
    train_targets = train_targets[-max_n_samples:, ...]
train_dataset = train_dataset / 255
train_dataset = train_dataset.reshape((train_dataset.shape[0],
                                       train_dataset.shape[1] * train_dataset.shape[2]))
encoder = OneHotEncoder(sparse=False)
train_targets = train_targets.reshape((train_targets.shape[0], 1))
train_targets = encoder.fit_transform(train_targets)

# From data frame to list
x_train, x_val, y_train, y_val, train_val_inds = array_to_list(
    input_data=train_dataset,
    output_data=train_targets)

# COMMAND ----------

# VAE Model constructor


def vae_model_constructor(latent_dim):

    # Sampling function
    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # Encoder
    encoder_inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Flatten()(x)
    x = Dense(16, activation="relu")(x)
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # Decoder
    latent_inputs = Input(shape=(latent_dim,))
    x = Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
    x = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
    decoder = Model(latent_inputs, decoder_outputs, name="decoder")

    class VAE(Model):
        def __init__(self, encoder, decoder, **kwargs):
            super(VAE, self).__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = Mean(name="total_loss")
            self.reconstruction_loss_tracker = Mean(
                name="reconstruction_loss"
            )
            self.kl_loss_tracker = Mean(name="kl_loss")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        binary_crossentropy(data, reconstruction), axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }

    # Create VAE model and compile it
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=Adam())

    return vae

# COMMAND ----------


# Training specs
train_specs = {'batch_size': batch_size}

# Hyper-parameter space
hyperparam_space = {'latent_dim': choice('i_n_layers', np.arange(1, 2))}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model_constructor=vae_model_constructor)

# COMMAND ----------

# Automatic Model Design
print('\n')
print('Automatic Model Design \n')
aironsuit.design(
    x_train=x_train,
    y_train=y_train,
    x_val=x_val,
    y_val=y_val,
    hyper_space=hyperparam_space,
    train_specs=train_specs,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    model_name=model_name,
    seed=0,
    patience=patience)
aironsuit.summary()
del x_train, x_val, y_train, y_val
