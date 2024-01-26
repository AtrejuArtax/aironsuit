import os
import pathlib

from utils import run_application

REPO_PATH = os.sep.join(str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-1])
APPLICATIONS = ["examples"]


def integration_test():
    # Test applications
    for name in APPLICATIONS:
        run_application(
            repo_path=REPO_PATH,
            application_name=name,
        )


if __name__ == "__main__":
    integration_test()
