import os
import os.path
from os.path import abspath

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
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
