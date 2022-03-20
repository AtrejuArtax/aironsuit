import argparse
import os
import pickle
import random
import warnings
from collections import Counter

import numpy as np
from hyperopt import Trials
from sklearn.metrics import classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from aironsuit.design.utils import choice_hp, uniform_hp
from aironsuit.suit import AIronSuit
from airontools.constructors.models.supervised.classification import ImageClassifierNN
from airontools.devices import get_available_gpus
from airontools.preprocessing import train_val_split

random.seed(0)
np.random.seed(0)
PROJECT = 'classification_pipeline'
EXECUTION_MODE = os.environ['EXECUTION_MODE'] if 'EXECUTION_MODE' in os.environ else 'development'
WORKING_PATH = os.path.join(os.path.expanduser("~"), 'airon', PROJECT, EXECUTION_MODE)


def image_classifier(input_shape, **reg_kwargs):

    # Create an image classification model and compile it
    classifier_nn = ImageClassifierNN(input_shape, **reg_kwargs)
    classifier_nn.compile(optimizer=Adam())

    return classifier_nn


def pipeline(new_design, design, max_n_samples, max_evals, epochs, batch_size, patience, verbose, precision):

    # Net name
    model_name = PROJECT + '_NN'

    # Data Pre-processing #

    # Load and preprocess data
    (train_dataset, train_targets), (test_dataset, test_targets) = mnist.load_data()
    if max_n_samples is not None:  # ToDo: test cases when max_n_samples is not None, like it is now it will crash
        train_dataset = train_dataset[-max_n_samples:, ...]
        train_targets = train_targets[-max_n_samples:, ...]
    train_dataset = np.expand_dims(train_dataset, -1) / 255
    test_dataset = np.expand_dims(test_dataset, -1) / 255
    train_targets = to_categorical(train_targets, 10)
    test_targets = to_categorical(test_targets, 10)
    data_specs = dict(input_shape=tuple(train_dataset.shape[1:]))

    # Sample weight
    sample_weight = np.ones((train_targets.shape[0], 1))
    counter = Counter(np.argmax(train_targets, axis=1).tolist())
    highest_count = np.max(list(counter.values()))
    for ind, count in counter.items():
        sample_weight[np.argmax(train_targets, axis=1) == ind] = highest_count / count

    # Automatic model design #

    # Model specs
    model_specs = dict()
    model_specs.update(data_specs)

    # Design
    aironsuit = None
    if design:

        # Split data per parallel model
        x_train, x_val, y_train, y_val, sample_weight_train, sample_weight_val, _ = train_val_split(
            input_data=train_dataset,
            output_data=train_targets,
            meta_data=sample_weight,
            return_tfrecord=False,  #ToDo: tfrecord compatible with airon predefined model classes (sub keras classes)
        )

        # Hyper-parameter space
        hyperparam_space = {
            'dropout_rate': uniform_hp('dropout_rate', 0., 0.4),
            'kernel_regularizer_l1': uniform_hp('kernel_regularizer_l1', 0., 0.001),
            'kernel_regularizer_l2': uniform_hp('kernel_regularizer_l2', 0., 0.001),
            'bias_regularizer_l1': uniform_hp('bias_regularizer_l1', 0., 0.001),
            'bias_regularizer_l2': uniform_hp('bias_regularizer_l2', 0., 0.001),
            'bn': choice_hp('bn', [True, False])
        }

        # Automatic model design
        print('\n')
        print('Automatic model design \n')
        trials_file_name = os.path.join(WORKING_PATH, 'design', 'trials.hyperopt')
        trials_exist = os.path.isfile(trials_file_name)
        if new_design or not trials_exist:
            trials = Trials()
        else:
            try:
                trials = pickle.load(open(trials_file_name, 'rb'))
            except RuntimeError as e:
                print(e)
                trials = Trials()
        aironsuit = AIronSuit(
            model_constructor=image_classifier,
            results_path=WORKING_PATH,
            name=model_name,
        )
        aironsuit.design(
            x_train=x_train,
            y_train=y_train,
            sample_weight=sample_weight_train,
            x_val=x_val,
            y_val=y_val,
            sample_weight_val=sample_weight_val,
            batch_size=batch_size,
            model_specs=model_specs,
            hyper_space=hyperparam_space,
            max_evals=max_evals + len(trials.trials),
            epochs=epochs,
            trials=trials,
            seed=0,
            patience=patience,
            verbose=verbose,
            optimise_hypers_on_the_fly=True,
        )
        del x_train, x_val, y_train, y_val
        aironsuit.summary()

    # Test Evaluation #

    if aironsuit is None:

        # Load aironsuit
        try:
            specs = model_specs.copy()
            best_file_name = os.path.join(WORKING_PATH, 'design', 'best_exp_' + model_name)
            with open(best_file_name + '_hparams', 'rb') as handle:
                specs.update(pickle.load(handle))
            aironsuit = AIronSuit(
                model_constructor=image_classifier,
                force_subclass_weights_saver=True,
                force_subclass_weights_loader=True
            )
            aironsuit.load_model(best_file_name, **specs)
        except RuntimeError as e:
            aironsuit = None
            print(e)

    if aironsuit is not None:

        # Split data per parallel model
        x_test, _, y_test, _, _ = train_val_split(
            input_data=test_dataset,
            output_data=test_targets
        )

        # Classification report
        y_pred = aironsuit.inference(x_test)
        test_report = classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        print('Evaluation report:')
        print(test_report)
        with open(WORKING_PATH + 'test_report', 'wb') as f:
            pickle.dump(test_report, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--h', action='store_true')
    parser.add_argument('--use_cpu', dest='use_cpu', action='store_true')
    parser.add_argument('--new_design', dest='new_design', default=False)
    parser.add_argument('--design', dest='design', default=True)
    parser.add_argument('--max_n_samples', dest='max_n_samples', type=int,
                        default=None if EXECUTION_MODE == 'production' else 1000)
    parser.add_argument('--max_evals', dest='max_evals', type=int, default=250 if EXECUTION_MODE == 'production' else 2)
    parser.add_argument('--epochs', dest='epochs', type=int, default=1000 if EXECUTION_MODE == 'production' else 25)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=32)
    parser.add_argument('--patience', dest='patience', type=int, default=5 if EXECUTION_MODE == 'production' else 2)
    parser.add_argument('--verbose', dest='verbose', type=int, default=0)
    parser.add_argument('--precision', dest='precision', type=str, default='float32')

    opts = parser.parse_args()
    print(''.join(f'{k}={v}\n' for k, v in vars(opts).items()))

    if opts.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    elif len(get_available_gpus()) == 0:
        warnings.warn('no gpus where found')
    del opts.h, opts.use_cpu

    pipeline(**vars(opts))
