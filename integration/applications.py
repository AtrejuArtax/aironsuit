import os
import pathlib
import shutil
import time

from utils import test_application

REPOS_PATH = os.sep.join(
    str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2]
)
APPLICATIONS = [[os.path.join("aironsuit", "examples"), ["tensorflow"]]]


def integration_test():
    # Test applications
    for app_name, app_backends in APPLICATIONS:
        test_application(REPOS_PATH, app_name)


if __name__ == "__main__":
    integration_test()
