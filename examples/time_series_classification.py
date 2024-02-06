import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from airontools.constructors.layers import layer_constructor
from airontools.path_utils import path_management
from airontools.preprocessing_utils import train_val_split
from hyperopt import Trials

from aironsuit.design.utils import choice_hp
from aironsuit.suit import AIronSuit

WORKING_PATH = os.path.expanduser("~")


def run_time_series_classification_example(working_dir: str) -> Tuple[float, float]:

    # Configuration
    example_name = "ensemble_mnist_example"
    model_name = "NN"
    working_path = os.path.join(working_dir, example_name)
    epochs = 3
    patience = 3
    max_evals = 3

    # Make/remove paths
    path_management(working_path, modes=["rm", "make"])

    # Load and preprocess data

    def readucr(filename):
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)

    root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
    x_train, y_train = readucr(root_url + "FordA_TRAIN.tsv")
    x_test, y_test = readucr(root_url + "FordA_TEST.tsv")
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    num_classes = len(np.unique(y_train))
    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Split data per parallel model
    x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
        input_data=x_train, output_data=y_train
    )

    # Classifier Model constructor
    model_specs = {
        "input_shape": x_train.shape[1:],
        "units": num_classes,
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"],
    }

    def classifier_model_constructor(**kwargs):
        if kwargs["num_heads"] == 0:
            num_heads = None
            sequential = True
        else:
            num_heads = kwargs["num_heads"]
            sequential = False
        classifier_kwargs = dict(
            filters=kwargs[
                "filters"
            ],  # Number of filters used for the convolutional layer
            kernel_size=kwargs[
                "kernel_size"
            ],  # Kernel size used for the convolutional layer
            strides=2,  # Strides used for the convolutional layer
            sequential_axis=1,  # Used to define the sequence for the self-attention or sequential layer
            num_heads=num_heads,  # Self-attention heads applied after the convolutional layer
            sequential=sequential,  # Whether to consider a sequential model or not
            units=kwargs["units"],  # Dense units applied after the self-attention layer
            activation="softmax",  # Output activation function
        )
        inputs = tf.keras.layers.Input(shape=kwargs["input_shape"])
        outputs = layer_constructor(x=inputs, **classifier_kwargs)
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            loss=kwargs["loss"],
            optimizer=kwargs["optimizer"],
            metrics=kwargs["metrics"],
        )

        return model

    # Hyper-parameter space
    hyperparam_space = {
        "filters": choice_hp("filters", [int(val) for val in np.arange(3, 30)]),
        "kernel_size": choice_hp("kernel_size", [int(val) for val in np.arange(3, 10)]),
        "num_heads": choice_hp("num_heads", [int(val) for val in np.arange(0, 10)]),
    }

    # Invoke AIronSuit
    aironsuit = AIronSuit(
        model_constructor=classifier_model_constructor, name=model_name
    )

    # Automatic Model Design
    print("\n")
    print("Automatic Model Design \n")
    aironsuit.design(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        model_specs=model_specs,
        hyper_space=hyperparam_space,
        max_evals=max_evals,
        epochs=epochs,
        trials=Trials(),
        seed=0,
        patience=patience,
        metric=0,
    )
    aironsuit.summary()

    # Evaluate
    score = aironsuit.model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save Model
    aironsuit.model.save_weights(os.path.join(working_path, model_name))
    best_hypers = aironsuit.load_hyper_candidates()
    del aironsuit

    # Re-Invoke AIronSuit and load model
    best_model_specs = model_specs.copy()
    best_model_specs.update(best_hypers)
    aironsuit = AIronSuit(
        model=classifier_model_constructor(**best_model_specs), name=model_name
    )
    aironsuit.model.load_weights(os.path.join(working_path, model_name))
    aironsuit.model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # Further Training
    aironsuit.train(epochs=epochs, x_train=x_train, y_train=y_train)

    # Evaluate
    loss, accuracy = aironsuit.model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    return loss, accuracy
