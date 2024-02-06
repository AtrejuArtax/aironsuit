import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from airontools.constructors.models.unsupervised.ae import AE
from airontools.path_utils import path_management
from airontools.preprocessing_utils import train_val_split
from hyperopt import Trials

from aironsuit.design.utils import choice_hp
from aironsuit.suit import AIronSuit

HOME = os.path.expanduser("~")


def run_ae_mnist_example(working_dir: str) -> float:

    # Configuration
    model_name = "AE_NN"
    working_path = os.path.join(working_dir, "airon", model_name)
    epochs = 10
    patience = 3
    max_evals = 3
    max_n_samples = 1000

    # Make/remove paths
    path_management(working_path, modes=["rm", "make"])

    # Load and preprocess data
    (train_dataset, target_dataset), _ = tf.keras.datasets.mnist.load_data()
    if max_n_samples is not None:
        train_dataset = train_dataset[-max_n_samples:, ...]
        target_dataset = target_dataset[-max_n_samples:, ...]
    train_dataset = np.expand_dims(train_dataset, -1) / 255
    train_dataset = train_dataset.reshape(
        (len(train_dataset), np.prod(list(train_dataset.shape)[1:]))
    )

    # Split data per parallel model
    x_train, x_val, _, meta_val, _ = train_val_split(
        input_data=train_dataset, meta_data=target_dataset
    )

    # AE Model constructor

    def ae_model_constructor(input_shape: Tuple[int], latent_dim: int):
        # Create AE model and compile it
        ae = AE(
            input_shape=input_shape,
            latent_dim=latent_dim,
        )
        ae.compile(optimizer=tf.keras.optimizers.Adam())

        return ae

    # Model specs
    model_specs = dict(
        input_shape=tuple(list(x_train.shape)[1:]),
    )

    # Hyper-parameter space
    hyperparam_space = dict(
        latent_dim=choice_hp("latent_dim", [int(val) for val in np.arange(3, 6)]),
    )

    # Invoke AIronSuit
    aironsuit = AIronSuit(
        model_constructor=ae_model_constructor,
        results_path=working_path,
        name=model_name,
    )

    # Automatic Model Design
    print("\n")
    print("Automatic Model Design \n")
    aironsuit.design(
        x_train=x_train,
        x_val=x_val,
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
    del x_train

    # Get latent insights
    aironsuit.visualize_representations(
        x_val,
        metadata=meta_val,
        hidden_layer_name="AE_z",
    )

    # Evaluate
    loss = float(aironsuit.model.evaluate(x_val)["loss"])
    print("Val loss:", loss)

    return loss


if __name__ == "__main__":
    run_ae_mnist_example(working_dir=HOME)
