import getopt
import os
import pathlib
import sys
import time
import shutil

from utils import test_application

REPOS_PATH = os.sep.join(str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2])
APPLICATIONS = [
    [os.path.join('aironsuit', 'examples'), ['tensorflow']]
]


def packages_manager(packages, mode):
    [os.system('pip {} {}'.format(mode, package)) for package in packages]
    time.sleep(10)


def integration_test():

    # Clear, build and install packages
    for package_name in ["ariontools", "aironsuit"]:
        repository_path = os.path.join(REPOS_PATH, package_name, os.sep)
        for name in ["dist", "build", package_name + ".egg-info"]:
            shutil.rmtree(os.path.join(repository_path, name))
        build_name = os.listdir(os.path.join(repository_path, "dist"))[0]
        os.system('python {}setup.py bdist_wheel'.format(build_name))
        for action, package_name_ in zip(["uninstall", "install"], [package_name, build_name]):
            os.system('pip {} {}'.format(action, package_name))

    # Test applications
    for app_name, app_backends in APPLICATIONS:
        test_application(REPOS_PATH, app_name)


if __name__ == '__main__':

    integration_test()
