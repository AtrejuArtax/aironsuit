import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()


setuptools.setup(
    name='aironsuit',
    version='0.1.0',
    scripts=[],
    author='Claudi Ruiz Camps',
    author_email='claudi_ruiz@hotmail.com',
    description='A model wrapper for hyper-parameter optimization and visualization purposes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/AtrejuArtax/aironsuit',
    packages=setuptools.find_packages(),
    classifiers=['Programming Language :: Python :: 3',
                 'License :: BSD 3-Clause',
                 'Operating System :: OS Independent'])
