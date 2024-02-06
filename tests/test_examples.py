import os
import os.path
from os.path import abspath

from examples.ae_mnist import run_ae_mnist_example
from examples.classification_mnist import run_classification_mnist_example
from examples.ensemble_mnist_classification import run_ensemble_mnist_example
from examples.standard_classification_pipeline import (
    run_standard_classification_pipeline_example,
)
from examples.time_series_classification import run_time_series_classification_example
from examples.variational_ae_mnist import run_vae_mnist_example

WORKING_DIR = os.sep.join(
    abspath(__file__).split(os.sep)[:-1] + ["test_generated_files", "examples"]
)


class TestExamples:

    def test_classification_mnist_example(
        self,
    ):
        """Test classification mnist example."""
        loss, accuracy = run_classification_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "classification_mnist_example"),
        )
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        #ToDo: fix performance inconsistency
        assert accuracy > 0.1

    def test_ae_mnist_example(
        self,
    ):
        """Test AE mnist example."""
        loss = run_ae_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "ae_mnist_example"),
        )
        assert isinstance(loss, float)
        assert loss < 0.3

    def test_ensemble_mnist_example(
        self,
    ):
        """Test ensemble mnist example."""
        accuracy = run_ensemble_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "ensemble_mnist_example"),
        )
        assert isinstance(accuracy, float)
        assert accuracy > 0.6

    def test_standard_classification_pipeline_example(
        self,
    ):
        """Test standard classification pipeline example."""
        accuracy = run_standard_classification_pipeline_example(
            working_dir=os.path.join(
                WORKING_DIR, "standard_classification_pipeline_example"
            ),
            max_n_samples=300,
            max_evals=2,
            epochs=2,
            patience=2,
        )
        assert isinstance(accuracy, float)
        assert accuracy > 0.6

    def test_time_series_classification_example(
        self,
    ):
        """Test time series classification example."""
        loss, accuracy = run_time_series_classification_example(
            working_dir=os.path.join(WORKING_DIR, "time_series_classification_example"),
        )
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert accuracy > 0.1

    def test_vae_mnist_example(
        self,
    ):
        """Test VAE mnist example."""
        loss = run_vae_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "vae_mnist_example"),
        )
        assert isinstance(loss, float)
        assert loss < 0.3
