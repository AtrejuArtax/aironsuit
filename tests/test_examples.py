import os
import os.path
from os.path import abspath

import pytest

from examples.ae_mnist_tf import ae_mnist_example
from examples.classification_mnist_tf import classification_mnist_example

WORKING_DIR = os.sep.join(
    abspath(__file__).split(os.sep)[:-1] + ["test_generated_files", "examples"]
)


class TestExamples:

    def test_classification_mnist_example(
        self,
    ):
        """Test classification mnist example."""
        loss, accuracy = classification_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "classification_mnist_example"),
        )
        # ToDo: fix classification mnist example accuracy and loss.
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert pytest.approx(loss, abs=0.04) == 2.3
        assert pytest.approx(accuracy, abs=0.04) == 0.113

    def test_ae_mnist_example(
        self,
    ):
        """Test AE mnist example."""
        loss = ae_mnist_example(
            working_dir=os.path.join(WORKING_DIR, "ae_mnist_example"),
        )
        assert isinstance(loss, float)
        assert pytest.approx(loss, abs=0.04) == 0.1201
