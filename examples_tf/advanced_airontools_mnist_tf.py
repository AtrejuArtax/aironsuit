# Databricks notebook source
import numpy as np
from hyperopt.hp import uniform, choice
from hyperopt import Trials
import random
import pickle
import os
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit
from airontools.constructors import model_constructor
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
from airontools.devices import get_available_gpus
random.seed(0)
np.random.seed(0)
HOME = os.path.expanduser("~")
OS_SEP = os.path.sep

# COMMAND ----------

# Example Set-Up #

project = 'mnist'
working_path = os.path.join(HOME, 'advanced_airontools_' + project)
use_gpu = True
max_n_samples = None
max_evals = 3
epochs = 100
batch_size = 32
patience = 3
parallel_models = 2
verbose = 0
precision = 'float32'

# COMMAND ----------

# Choose devices
if not use_gpu or len(get_available_gpus()) == 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    devices = ['/cpu:0']
else:
    devices = [gpu_name.replace('/device:GPU:', '/gpu:') for gpu_name in get_available_gpus()]

# Net name
model_name = project + '_NN'

# Paths
prep_data_path = os.path.join(working_path, 'PrepDatasets')
inference_data_path = os.path.join(working_path, 'Inference')
results_path = os.path.join(working_path, 'Results')

# Target names
target_names = [str(i) for i in range(10)]

# Input/Output Specs
metric = 'categorical_accuracy' if project in ['fashion_mnist', 'mnist'] else None
input_specs = {'image': {'type': 'image', 'sequential': False, 'dim': 784}}
output_specs = {'y': {'type': 'cat', 'sequential': False, 'dim': len(target_names)}}
data_specs = {'input_specs': input_specs,
              'output_specs': output_specs}

# Training specs
train_specs = {
    'batch_size': batch_size,
    'path': results_path}

# Model Specs
model_specs = {
    'name': model_name,
    'input_specs': input_specs,
    'output_specs': output_specs,
    'bn': True,
    'devices': devices,
    'parallel_models': parallel_models,
    'precision': precision,
    'sequential_block': False,
    'optimizer': Adam(learning_rate=0.001),
    'hidden_activation': 'prelu',
    'output_activation': 'softmax'}
hyperparam_space = {
    'dropout_rate': uniform('dropout_rate', 0., 0.4),
    'kernel_regularizer_l1': uniform('kernel_regularizer_l1', 0., 0.001),
    'kernel_regularizer_l2': uniform('kernel_regularizer_l2', 0., 0.001),
    'bias_regularizer_l1': uniform('bias_regularizer_l1', 0., 0.001),
    'bias_regularizer_l2': uniform('bias_regularizer_l2', 0., 0.001),
    'compression': uniform('compression', 0.3, 0.98),
    'i_n_layers': choice('i_n_layers', np.arange(1, 2)),
    'c_n_layers': choice('c_n_layers', np.arange(1, 2))}
hyperparam_space.update({'loss': choice('loss', ['mse', 'categorical_crossentropy'])})

# Make/remove paths
path_modes = ['rm', 'make']
for path in [prep_data_path, inference_data_path, results_path]:
    path_management(path, modes=path_modes)

# COMMAND ----------

# Load and preprocess data
(train_dataset, train_targets), _ = mnist.load_data()
if max_n_samples is not None:
    train_dataset = train_dataset[-max_n_samples:, ...]
    train_targets = train_targets[-max_n_samples:, ...]
train_dataset = train_dataset.astype(precision) / 255
train_dataset = train_dataset.reshape((train_dataset.shape[0],
                                       train_dataset.shape[1] * train_dataset.shape[2]))
encoder = OneHotEncoder(sparse=False)
train_targets = train_targets.reshape((train_targets.shape[0], 1))
train_targets = encoder.fit_transform(train_targets)

# From data frame to list
x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
    input_data=train_dataset,
    output_data=train_targets,
    n_parallel_models=model_specs['parallel_models'],
    do_kfolds=False if model_specs['parallel_models'] == 1 else True)

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model_constructor=model_constructor)

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
    model_specs=model_specs,
    train_specs=train_specs,
    path=results_path,
    max_evals=max_evals,
    epochs=epochs,
    trials=Trials(),
    name=model_name,
    verbose=verbose,
    seed=0,
    metric=metric,
    val_inference_in_path=results_path,
    patience=patience)
aironsuit.summary()
del x_train, x_val, y_train, y_val

# COMMAND ----------

# Test Evaluation #

# Load and preprocess data
(_, train_targets), (test_dataset, test_targets) = mnist.load_data()
test_dataset = test_dataset / 255
test_dataset = test_dataset.reshape((test_dataset.shape[0], test_dataset.shape[1] * test_dataset.shape[2]))
encoder = OneHotEncoder(sparse=False)
train_targets = train_targets.reshape((train_targets.shape[0], 1))
test_targets = test_targets.reshape((test_targets.shape[0], 1))
encoder.fit(train_targets)
test_targets = encoder.transform(test_targets)

# From data frame to list
x_test, _, y_test, _, _ = train_val_split(
    input_data=test_dataset,
    output_data=test_targets,
    n_parallel_models=model_specs['parallel_models'] * len(model_specs['devices']),
    val_ratio=0)
y_test = y_test[0]

# Inference
y_pred = aironsuit.inference(x_test)
y_pred = y_pred if not isinstance(y_pred, list) else np.mean([y_pred_ for y_pred_ in y_pred], axis=0)

# Classification report
test_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
print(test_report)
with open(results_path + 'test_report', 'wb') as f:
    pickle.dump(test_report, f, protocol=pickle.HIGHEST_PROTOCOL)
