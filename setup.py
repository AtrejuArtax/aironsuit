import os

import setuptools

PACKAGE_NAME = 'aironsuit'

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name=PACKAGE_NAME,
    version='0.1.19',
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
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'hyperopt',
        'tensorflow==2.7.0',
        'tensorboard==2.7.0',
        'tensorflow_datasets',
        'airontools==0.1.16'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent'
    ],
    license='BSD')
