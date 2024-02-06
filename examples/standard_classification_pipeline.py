import os
import pickle
import random
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from airontools.constructors.models.supervised.image_classifier import ImageClassifierNN
from airontools.preprocessing_utils import train_val_split
from hyperopt import Trials
from sklearn.metrics import classification_report

from aironsuit.design.utils import choice_hp, uniform_hp
from aironsuit.suit import AIronSuit

WORKING_PATH = os.path.expanduser("~")


def image_classifier(input_shape: Tuple[None, int], n_classes: int, **kwargs):
    # Create an image classification model and compile it
    classifier_nn = ImageClassifierNN(
        input_shape=input_shape,
        n_classes=n_classes,
        **kwargs,
    )
    classifier_nn.compile(optimizer=tf.keras.optimizers.Adam())

    return classifier_nn


def run_standard_classification_pipeline_example(
    working_dir: str,
    new_design: Optional[bool] = True,
    design: Optional[bool] = True,
    max_n_samples: Optional[int] = None,
    max_evals: Optional[int] = 250,
    epochs: Optional[int] = 1000,
    batch_size: Optional[int] = 32,
    patience: Optional[int] = 3,
    verbose: Optional[int] = 0,
) -> Optional[float]:

    random.seed(0)
    np.random.seed(0)

    # Configuration
    example_name = "standard_classification_pipeline_example"
    model_name = "NN"
    working_path = os.path.join(working_dir, example_name)

    # Data Pre-processing #

    # Load and preprocess data
    (train_dataset, train_targets), (
        test_dataset,
        test_targets,
    ) = tf.keras.datasets.mnist.load_data()
    if (
        max_n_samples is not None
    ):  # ToDo: test cases when max_n_samples is not None, like it is now it will crash
        train_dataset = train_dataset[-max_n_samples:, ...]
        train_targets = train_targets[-max_n_samples:, ...]
    train_dataset = np.expand_dims(train_dataset, -1) / 255
    test_dataset = np.expand_dims(test_dataset, -1) / 255
    train_targets = tf.keras.utils.to_categorical(train_targets, 10)
    test_targets = tf.keras.utils.to_categorical(test_targets, 10)
    data_specs = dict(
        input_shape=tuple(train_dataset.shape[1:]),
        n_classes=train_targets.shape[-1],
    )

    # Sample weight
    sample_weight = np.ones((train_targets.shape[0], 1))
    counter = Counter(np.argmax(train_targets, axis=1).tolist())
    highest_count = np.max(list(counter.values()))
    for ind, count in counter.items():
        sample_weight[np.argmax(train_targets, axis=1) == ind] = highest_count / count

    # Automatic model design #

    # Model specs
    model_specs = dict(
        filters=32,
        kernel_size=15,
        strides=2,
        sequential_axis=-1,
        num_heads=3,
    )
    model_specs.update(data_specs)

    # Design
    aironsuit = None
    if design:
        # Split data per parallel model
        (
            x_train,
            x_val,
            y_train,
            y_val,
            sample_weight_train,
            sample_weight_val,
            _,
        ) = train_val_split(
            input_data=train_dataset,
            output_data=train_targets,
            meta_data=sample_weight,
            return_tfrecord=False,  # ToDo: tfrecord compatible with airon predefined model classes (sub keras classes)
        )

        # Hyper-parameter space
        hyperparam_space = {
            "dropout_rate": uniform_hp("dropout_rate", 0.0, 0.4),
            "kernel_regularizer_l1": uniform_hp("kernel_regularizer_l1", 0.0, 0.001),
            "kernel_regularizer_l2": uniform_hp("kernel_regularizer_l2", 0.0, 0.001),
            "bias_regularizer_l1": uniform_hp("bias_regularizer_l1", 0.0, 0.001),
            "bias_regularizer_l2": uniform_hp("bias_regularizer_l2", 0.0, 0.001),
            "bn": choice_hp("bn", [True, False]),
        }

        # Automatic model design
        print("\n")
        print("Automatic model design \n")
        trials_file_name = os.path.join(working_path, "trials.hyperopt")
        trials_exist = os.path.isfile(trials_file_name)
        if new_design or not trials_exist:
            trials = Trials()
        else:
            try:
                trials = pickle.load(open(trials_file_name, "rb"))
            except RuntimeError as e:
                print(e)
                trials = Trials()
        aironsuit = AIronSuit(
            model_constructor=image_classifier,
            results_path=working_path,
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
            metric="loss",
        )
        del x_train, x_val, y_train, y_val
        aironsuit.summary()

    # Test Evaluation #

    if aironsuit is None:
        # Load aironsuit
        try:
            specs = model_specs.copy()
            best_file_name = os.path.join(
                working_path, "design", "best_exp_" + model_name
            )
            with open(best_file_name + "_hparams", "rb") as handle:
                specs.update(pickle.load(handle))
            aironsuit = AIronSuit(
                model=image_classifier(**specs),
                name=model_name,
            )
        except RuntimeError as e:
            aironsuit = None
            print(e)

    if aironsuit is not None:
        # Split data per parallel model
        x_test, _, y_test, _, _ = train_val_split(
            input_data=test_dataset, output_data=test_targets
        )

        # Classification report
        y_pred = aironsuit.inference(x_test)
        test_report = classification_report(
            np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1)
        )
        print("Evaluation report:")
        print(test_report)
        with open(working_path + "test_report", "wb") as f:
            pickle.dump(test_report, f, protocol=pickle.HIGHEST_PROTOCOL)
        accuracy = test_report["accuracy"]

        return accuracy


if __name__ == "__main__":
    run_standard_classification_pipeline_example(working_dir=WORKING_PATH)
