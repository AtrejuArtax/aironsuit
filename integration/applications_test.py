import os
import pathlib
import shutil
import time

from utils import test_application

# Available tests: integration, performance
EXECUTION_MODE = os.environ["EXECUTION_MODE"] if "EXECUTION_MODE" in os.environ else "integration"
REPOS_PATH = os.sep.join(str(pathlib.Path(__file__).parent.resolve()).split(os.sep)[:-2])
APPLICATIONS = [
    [os.path.join('aironsuit', 'examples'), ['tensorflow']]
]


def packages_manager(packages, mode):
    [os.system('pip {} {}'.format(mode, package)) for package in packages]
    time.sleep(10)


def applications_test():

    print(EXECUTION_MODE.upper(), "TEST \n")

    # Clear, build and install packages
    for package_name in ["airontools", "aironsuit"]:
        repository_path = os.path.join(REPOS_PATH, package_name) + os.sep
        for name in ["dist", "build", package_name + ".egg-info"]:
            name_path = os.path.join(repository_path, name)
            if os.path.isdir(name_path):
                shutil.rmtree(name_path)
        os.system('cd {} && python setup.py bdist_wheel'.format(repository_path))
        build_name = os.listdir(os.path.join(repository_path, "dist"))[0]
        for action, package_name_ in zip(["uninstall", "install"], [package_name, build_name]):
            os.system('pip {} {}'.format(action, package_name))

    # Test applications
    for app_name, app_backends in APPLICATIONS:
        test_application(
            repos_path=REPOS_PATH,
            application_name=app_name,
            execution_mode=EXECUTION_MODE
        )


if __name__ == '__main__':

    applications_test()
