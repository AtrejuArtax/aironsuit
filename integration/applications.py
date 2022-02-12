import getopt
import os
import sys
import time

from utils import test_application

BACKENDS = ['tensorflow']
REPOS_PATH = os.path.join(os.path.expanduser('~'), 'repositories')
APPLICATIONS = [
    [os.path.join('aironsuit', 'examples'), ['tensorflow']]
]
TEST_PACKAGES = ['airontools', 'aironsuit']
OTHER_TF_PACKAGES = ['tensorboard']


def packages_manager(packages, mode):
    if mode == 'install':
        arguments = '--force-reinstall'
    else:
        arguments = '-y'
    [os.system('pip {} {} {}'.format(mode, arguments, package)) for package in packages]
    time.sleep(10)


def integration_test(test_version, quick_test):

    # Local test packages
    local_test_packages = [
        os.path.join(REPOS_PATH, 'airontools', 'dist', 'airontools-' + test_version + '-py3-none-any.whl'),
        os.path.join(REPOS_PATH, 'aironsuit', 'dist', 'aironsuit-' + test_version + '-py3-none-any.whl')]

    # Test applications
    installed_backends = ['tensorflow', 'pytorch-lightning']
    if not quick_test:
        packages_manager(installed_backends + TEST_PACKAGES, 'uninstall')
    for app_name, app_backends in APPLICATIONS:
        if not quick_test:
            install_packages = local_test_packages + app_backends
            if 'tensorflow' in app_backends:
                install_packages += OTHER_TF_PACKAGES
            packages_manager(install_packages, 'install')
        test_application(app_name)
        if not quick_test:
            uninstall_packages = app_backends + TEST_PACKAGES
            if 'tensorflow' in app_backends:
                uninstall_packages += OTHER_TF_PACKAGES
            packages_manager(uninstall_packages, 'uninstall')
    if not quick_test:
        packages_manager(installed_backends, 'install')


if __name__ == '__main__':

    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, 'h', ['version=', 'quick_test='])
    except getopt.GetoptError:
        sys.exit(2)

    version = '0.1.14'
    quick = True

    for opt, arg in opts:

        print('\n')
        if opt == '-h':
            sys.exit()
        if opt in '--version':
            version = arg
            print('version:' + arg)
        if opt in '--quick':
            quick = arg
            print('quick:' + arg)

    integration_test(version, quick)
