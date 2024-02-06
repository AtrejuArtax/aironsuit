import os
import random
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


def run_classification_mnist_example(working_dir: str) -> Tuple[float, float]:

    random.seed(0)
    np.random.seed(0)

    # Configuration
    example_name = "classification_mnist_example"
    model_name = "NN"
    working_path = os.path.join(working_dir, example_name)
    num_classes = 10
    epochs = 2
    patience = 2
    max_evals = 2

    # Make/remove paths
    path_management(working_path, modes=["rm", "make"])

    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255, -1)
    x_test = np.expand_dims(x_test.astype("float32") / 255, -1)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Split data per parallel model
    x_train, x_val, y_train, y_val, train_val_inds = train_val_split(
        input_data=x_train,
        output_data=y_train,
        return_tfrecord=False,
    )

    # Classifier Model constructor
    model_specs = {
        "input_shape": (28, 28, 1),
        "loss": "categorical_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"],
    }

    def classifier_model_constructor(**kwargs):
        inputs = tf.keras.layers.Input(shape=kwargs["input_shape"])
        outputs = layer_constructor(
            x=inputs,
            filters=kwargs[
                "filters"
            ],  # Number of filters used for the convolutional layer
            kernel_size=(
                kwargs["kernel_size"],
                kwargs["kernel_size"],
            ),  # Kernel size used for the convolutional layer
            strides=2,  # Strides used for the convolutional layer
            sequential_axis=-1,  # Channel axis, used to define the sequence for the self-attention layer
            num_heads=kwargs[
                "num_heads"
            ],  # Self-attention heads applied after the convolutional layer
            units=10,  # Dense units applied after the self-attention layer
            activation="softmax",  # Output activation function
        )
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
        "num_heads": choice_hp("num_heads", [int(val) for val in np.arange(2, 10)]),
    }

    # Invoke AIronSuit
    aironsuit = AIronSuit(
        model_constructor=classifier_model_constructor,
        name=model_name,
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
        metric="loss",
    )
    aironsuit.summary()

    # Evaluate
    score = aironsuit.model.evaluate(x_test, y_test)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Save Model
    aironsuit.model.save_weights(os.path.join(working_path, model_name))
    best_model_specs = model_specs.copy()
    best_model_specs.update(aironsuit.load_hyper_candidates())
    del aironsuit

    # Re-Invoke AIronSuit and load model
    aironsuit = AIronSuit(
        model=classifier_model_constructor(**best_model_specs),
        name=model_name,
    )
    aironsuit.model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    aironsuit.model.load_weights(os.path.join(working_path, model_name))

    # Further Training
    aironsuit.train(
        epochs=epochs,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
    )

    # Evaluate
    loss, accuracy = aironsuit.model.evaluate(x_test, y_test)
    print("Test loss:", loss)
    print("Test accuracy:", accuracy)

    return loss, accuracy


if __name__ == "__main__":
    run_classification_mnist_example(working_dir=WORKING_PATH)
