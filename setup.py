import setuptools
import os

PACKAGE_NAME = 'aironsuit'

with open('README.md', 'r') as fh:
    long_description = fh.read()

if 'AIRONSUIT_BACKEND' in os.environ:
    EXTRA_PKGS = [os.environ['AIRONSUIT_BACKEND']]
else:
    EXTRA_PKGS = ['tensorflow', 'torch']


setuptools.setup(
    name=PACKAGE_NAME,
    version='0.1.2',
    scripts=[],
    author='Claudi Ruiz Camps',
    author_email='claudi_ruiz@hotmail.com',
    description='A model wrapper for automatic model design and visualization purposes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AtrejuArtax/aironsuit',
    packages=setuptools.find_packages(
        include=[PACKAGE_NAME] + [PACKAGE_NAME + '.' + name
                                  for name in os.listdir(os.path.join(os.getcwd(), PACKAGE_NAME))
                                  if not any([str_ in name for str_ in ['.py', '__']])]),
    install_requires=['sklearn', 'hyperopt'] + EXTRA_PKGS,
    classifiers=['Programming Language :: Python :: 3',
                 'License :: OSI Approved :: BSD License',
                 'Operating System :: OS Independent'],
    license='BSD')
