# AIronSuit

AIronSuit (Beta) is a Python library for automatic model design/selection and visualization purposes built to work with 
[tensorflow](https://github.com/tensorflow/tensorflow) (or [pytorch](https://github.com/pytorch/pytorch) in the future) 
as a backend. It aims to accelerate
the development of deep learning approaches for research/development purposes by providing components relying on cutting 
edge approaches. It is flexible and its components can be 
replaced by customized ones from the user. The user mostly focuses on defining the input and output, 
and AIronSuit takes care of its optimal mapping. 

Key features:

1. Automatic model design/selection with [hyperopt](https://github.com/hyperopt/hyperopt). 
2. Parallel computing for multiple models across multiple GPUs when using a k-fold approach.
3. Built-in model trainer that saves training progression to be visualized with 
   [TensorBoard](https://github.com/tensorflow/tensorboard).
4. Machine learning tools from [AIronTools](https://github.com/AtrejuArtax/airontools): `net_constructor`, `custom_block`, 
   `custom_layer`, preprocessing utils, etc.
5. Flexibility: the user can replace AIronSuit components by a user customized one. For instance,
    the net constructor can be easily replaced by a user customized one.
   
### Installation

`pip install aironsuit`

### Examples

see usage examples in [aironsuit/examples](https://github.com/AtrejuArtax/aironsuit/tree/master/examples)