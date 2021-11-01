# Databricks notebook source
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, Reshape
from tensorflow.keras.metrics import Mean
from tensorflow.keras.losses import binary_crossentropy, mse
from tensorflow.keras.optimizers import Adam
from hyperopt.hp import choice
from hyperopt import Trials
import os
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.preprocessing import train_val_split
from airontools.model_constructors import layer_constructor

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

# From data frame to list
x_train, x_val, train_val_inds = train_val_split(input_data=train_dataset)

# COMMAND ----------

# VAE Model constructor


def vae_model_constructor(latent_dim):

    # Sampling layer
    class Sampling(Layer):
        """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # VAE class

    class VAE(Model):
        def __init__(self, latent_dim, **kwargs):
            super(VAE, self).__init__(**kwargs)

            self.total_loss_tracker = Mean(name="total_loss")
            self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
            self.kl_loss_tracker = Mean(name="kl_loss")

            # Encoder
            encoder_inputs = Input(shape=(28, 28, 1))
            encoder_conv = layer_constructor(
                encoder_inputs, name='encoder_conv', filters=32, kernel_size=3, strides=2, advanced_reg=True)
            z_mean = layer_constructor(encoder_conv, name='encoder_mean', units=latent_dim, advanced_reg=True)
            z_log_var = layer_constructor(encoder_conv, name='encoder_log_var', units=latent_dim, advanced_reg=True)
            z = Sampling()([z_mean, z_log_var])
            self.encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

            # Decoder
            latent_inputs = Input(shape=(latent_dim,))
            decoder_outputs = layer_constructor(latent_inputs, name='encoder_dense', units=7 * 7 * 64, advanced_reg=True)
            decoder_outputs = Reshape((7, 7, 64))(decoder_outputs)
            for i, filters, activation in zip([1, 2], [64, 32], ['relu', 'relu']):
                decoder_outputs = layer_constructor(
                    decoder_outputs,
                    name='decoder_conv', name_ext=str(i), activation=activation, filters=filters, kernel_size=3,
                    strides=2, padding='same', conv_transpose=True, advanced_reg=True)
            decoder_outputs = layer_constructor(
                decoder_outputs,
                name='decoder_output', activation='sigmoid', filters=1, kernel_size=3, padding='same',
                conv_transpose=True, advanced_reg=True)
            self.decoder = Model(latent_inputs, decoder_outputs, name="decoder")

        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]

        def train_step(self, data):
            total_loss, reconstruction_loss, kl_loss, tape = self.loss_evaluation(data, return_tape=True)
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

        def evaluate(self, data):
            return self.loss_evaluation(data)[0]

        def loss_evaluation(self, data, return_tape=False):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        binary_crossentropy(data[:], reconstruction[:]), axis=(1, 2)
                    )
                )
                kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
                kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
                total_loss = reconstruction_loss + kl_loss
            returns = [total_loss, reconstruction_loss, kl_loss]
            if return_tape:
                returns += [tape]
            return returns

        def save_weights(self, path):
            with open(path + '_encoder', 'w') as f:
                json.dump([w.tolist() for w in self.encoder.get_weights()], f)
            with open(path + '_decoder', 'w') as f:
                json.dump([w.tolist() for w in self.decoder.get_weights()], f)

        def load_weights(self, path):
            with open(path + '_encoder', 'r') as f:
                encoder_weights = [np.array(w) for w in json.load(f)]
            self.encoder.set_weights(encoder_weights)
            with open(path + '_decoder', 'r') as f:
                decoder_weights = [np.array(w) for w in json.load(f)]
            self.decoder.set_weights(decoder_weights)

        def summary(self):
            self.encoder.summary()
            self.decoder.summary()

    # Create VAE model and compile it
    vae = VAE(latent_dim)
    vae.compile(optimizer=Adam())

    return vae

# COMMAND ----------


# Training specs
train_specs = {'batch_size': batch_size}

# Hyper-parameter space
hyperparam_space = {'latent_dim': choice('latent_dim', np.arange(2, 6))}

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model_constructor=vae_model_constructor,
                      force_subclass_weights_saver=True,
                      force_subclass_weights_loader=True)

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
    patience=patience)
aironsuit.summary()
del x_train, x_val
