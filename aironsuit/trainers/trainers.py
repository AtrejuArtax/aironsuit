from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.trainers.trainers_tf import *
    from tensorflow.keras.models import Model as Module
else:
    from aironsuit.trainers.trainers_torch import *
    from torch.nn import Module


class AIronTrainer(object):
    """ AIronTrainer is a module wrapper that takes care of the module training and prediction.

        Attributes:
            module (Module): NN module.
            best_module_name (str): Best module name.
            __class_weight (dict): Weight per class when performing a classification task.
            __path (str): Path where to save intermediate optimizations.

    """

    def __init__(self, module, **kwargs):
        available_kwargs = ['callbacks', 'mode', 'class_weight', 'path', 'batch_size']
        assert all([kwarg in available_kwargs for kwarg in kwargs.keys()])
        self.module = module
        self.best_module_name = None
        self.__class_weight = kwargs['class_weight'] if 'class_weight' in kwargs else None
        self.__path = kwargs['path'] if 'path' in kwargs else None

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def fit(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        """ Module for fitting.

            Parameters:
                x_train (list, np.array): Input data for training.
                y_train (list, np.array): Output data for training.
                x_val (list, np.array): Input data for validation.
                y_val (list, np.array): Output data for validation.
        """
        fit(module=self.module, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, **kwargs)

    def evaluate(self, x_val, y_val):
        """ Module for evaluation.

            Parameters:
                x_val (list, np.array): Input data for validation.
                y_val (list, np.array): Output data for validation.
        """
        evaluate(module=self.module, x_val=x_val, y_val=y_val)

    def predict(self, x, **kwargs):
        """ Module prediction.

            Parameters:
                x (list, np.array): Input data for prediction.
        """
        return self.module.predict(x, **kwargs)
