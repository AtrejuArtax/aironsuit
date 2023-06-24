import os
import pathlib
import time

from utils import test_application

REPOS_PATH = os.sep.join(
    str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2]
)
APPLICATIONS = [[os.path.join("aironsuit", "examples"), ["tensorflow"]]]


def packages_manager(packages, mode):
    [os.system("pip {} {}".format(mode, package)) for package in packages]
    time.sleep(10)


def integration_test():
    # Test applications
    for app_name, app_backends in APPLICATIONS:
        test_application(REPOS_PATH, app_name)


if __name__ == "__main__":
    integration_test()
