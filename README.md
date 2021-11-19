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
4. Machine learning tools from [AIronTools](https://github.com/AtrejuArtax/airontools): `model_constructor`, `block_constructor`, 
   `layer_constructor`, preprocessing utils, etc.
5. Flexibility: the user can replace AIronSuit components by a customized one. For instance,
    the model constructor can be easily replaced by a customized one.
   
### Installation

`pip install aironsuit`

### Example

``` python
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
from airontools.tools import path_management
HOME = os.path.expanduser("~")
OS_SEP = os.path.sep

# COMMAND ----------

# Example Set-Up #

model_name = 'VAE_NN'
working_path = os.path.join(HOME, 'airon', model_name) + OS_SEP
num_classes = 10
batch_size = 128
epochs = 30
patience = 3
max_evals = 3
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
    force_subclass_weights_loader=True,
    path=working_path
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
```
![alt text](https://github.com/AtrejuArtax/aironsuit/blob/visualization/vae_z_representations.png?raw=true)

### More Examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)