import os
import sys
import pickle
import getopt
import random
import numpy as np
from hyperopt import Trials
from hyperopt.hp import choice, uniform
from sklearn.metrics import classification_report
from tensorflow.python.client import device_lib
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
os.environ['AIRONSUIT_BACKEND'] = 'tensorflow'
from aironsuit.suit import AIronSuit  # aironsuit==0.1.9, airontools==0.1.9
from airontools.preprocessing import train_val_split
from airontools.tools import path_management
from airontools.constructors.models.supervised.classification import ImageClassifierNN
random.seed(0)
np.random.seed(0)
OS_SEP = os.path.sep
PROJECT = 'classification_pipeline'
EXECUTION_MODE = os.environ['EXECUTION_MODE'] if 'EXECUTION_MODE' in os.environ else 'development'


def image_classifier(input_shape, **reg_kwargs):

    # Create an image classification model and compile it
    classifier_nn = ImageClassifierNN(input_shape, **reg_kwargs)
    classifier_nn.compile(optimizer=Adam())

    return classifier_nn


def pipeline():

    # Net name
    model_name = PROJECT + '_NN'

    # Paths
    prep_data_path = os.path.join(working_path, 'PrepDatasets' + OS_SEP)
    inference_data_path = os.path.join(working_path, 'Inference' + OS_SEP)
    results_path = os.path.join(working_path, 'Results' + OS_SEP)
    for path in [prep_data_path, inference_data_path, results_path]:
        path_management(path)

    # Data Pre-processing #

    # Load and preprocess data
    (train_dataset, train_targets), (test_dataset, test_targets) = mnist.load_data()
    if max_n_samples is not None:  # ToDo: test cases when max_n_samples is not None
        train_dataset = train_dataset[-max_n_samples:, ...]
        train_targets = train_targets[-max_n_samples:, ...]
    train_dataset = np.expand_dims(train_dataset, -1).astype(precision) / 255
    test_dataset = np.expand_dims(test_dataset, -1).astype(precision) / 255
    train_targets = to_categorical(train_targets, 10)
    test_targets = to_categorical(test_targets, 10)
    data_specs = dict(input_shape=tuple(train_dataset.shape[1:]))

    # Automatic model design #

    # Model specs
    model_specs = dict()
    model_specs.update(data_specs)

    aironsuit = None
    if explore:

        # From data frame to list
        x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
            input_data=train_dataset,
            output_data=train_targets)

        # Training specs
        train_specs = {'batch_size': batch_size}

        # Hyper-parameter space
        hyperparam_space = {
            'dropout_rate': uniform('dropout_rate', 0., 0.4),
            'kernel_regularizer_l1': uniform('kernel_regularizer_l1', 0., 0.001),
            'kernel_regularizer_l2': uniform('kernel_regularizer_l2', 0., 0.001),
            'bias_regularizer_l1': uniform('bias_regularizer_l1', 0., 0.001),
            'bias_regularizer_l2': uniform('bias_regularizer_l2', 0., 0.001),
            'bn': choice('bn', [True, False])}

        # Automatic model design
        print('\n')
        print('Automatic model design \n')
        trials = None
        if new_exploration:
            trials = Trials()
        elif os.path.isfile(results_path + 'trials.hyperopt'):
            try:
                trials = pickle.load(open(results_path + 'trials.hyperopt', 'rb'))
            except RuntimeError as e:
                print(e)
        aironsuit = AIronSuit(
            model_constructor=image_classifier,
            force_subclass_weights_saver=True,
            force_subclass_weights_loader=True)
        aironsuit.design(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            model_specs=model_specs,
            hyper_space=hyperparam_space,
            train_specs=train_specs,
            max_evals=max_evals,
            epochs=epochs,
            trials=trials,
            name=model_name,
            seed=0,
            patience=patience)
        del x_train, x_val, y_train, y_val
        aironsuit.summary()

    # Test Evaluation #

    if aironsuit is None:

        # Load aironsuit
        try:
            specs = model_specs.copy()
            with open(results_path + 'best_exp_' + model_name + '_hparams', 'rb') as handle:
                specs.update(pickle.load(handle))
            aironsuit = AIronSuit(
                model_constructor=image_classifier,
                force_subclass_weights_saver=True,
                force_subclass_weights_loader=True)
            aironsuit.load_model(results_path + 'best_exp_' + model_name, **specs)
        except RuntimeError as e:
            aironsuit = None
            print(e)

    if aironsuit is not None:

        # From data frame to list
        x_test, _, y_test, _, _ = train_val_split(
            input_data=test_dataset,
            output_data=test_targets)

        # Classification report
        y_pred = aironsuit.inference(x_test)
        test_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print('Evaluation report:')
        print(test_report)
        with open(results_path + 'test_report', 'wb') as f:
            pickle.dump(test_report, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, 'h', [
            'working_path=',
            'new_preprocessing=',
            'new_exploration=',
            'explore=',
            'use_gpu=',
            'max_evals=',
            'epochs=',
            'batch_size=',
            'patience=',
            'verbose=',
            'precision='])
    except getopt.GetoptError:
        sys.exit(2)

    working_path = os.path.join(os.path.expanduser('~'), PROJECT)
    new_exploration = True
    explore = True
    use_gpu = True
    max_n_samples = None
    max_evals = 1000 if EXECUTION_MODE == 'production' else 3
    epochs = 1000 if EXECUTION_MODE == 'production' else 2
    batch_size = 32
    patience = 5 if EXECUTION_MODE == 'production' else 3
    verbose = 0
    precision = 'float32'

    for opt, arg in opts:

        print('\n')
        if opt == '-h':
            sys.exit()
        if opt in '--working_path':
            working_path = arg
            print('working_path:' + arg)
        elif opt in '--new_exploration':
            new_exploration = arg == 'True'
            print('new_exploration:' + arg)
        elif opt in '--explore':
            explore = arg == 'True'
            print('explore:' + arg)
        elif opt in '--use_gpu':
            use_gpu = arg == 'True'
            print('use_gpu:' + arg)
        elif opt in '--max_n_samples':
            max_n_samples = int(arg) if arg != 'None' else None
            print('max_n_samples:' + arg)
        elif opt in '--max_evals':
            max_evals = int(arg)
            print('max_evals:' + arg)
        elif opt in '--epochs':
            epochs = int(arg)
            print('epochs:' + arg)
        elif opt in '--batch_size':
            batch_size = int(arg)
            print('batch_size:' + arg)
        elif opt in '--patience':
            patience = int(arg)
            print('patience:' + arg)
        elif opt in '--verbose':
            verbose = int(arg)
            print('verbose:' + arg)
        elif opt in '--precision':
            precision = arg
            print('precision:' + arg)

    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

    if not use_gpu or len(get_available_gpus()) == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
        devices = ['/cpu:0']
    else:
        devices = [gpu_name.replace('/device:GPU:', '/gpu:') for gpu_name in get_available_gpus()]

    pipeline()
