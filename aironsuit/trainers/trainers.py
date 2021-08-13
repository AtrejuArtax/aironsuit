import os
import glob
from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.trainers.trainers_tf import *
    from tensorflow.keras.models import Model
else:
    from aironsuit.trainers.trainers_torch import *
    from torch.nn import Module as Model


class AIronTrainer(object):
    """ AIronTrainer is a model wrapper that takes care of the model training and prediction.

        Attributes:
            model (Model): NN model.
            best_model_name (str): Best model name.
            __class_weight (dict): Weight per class when performing a classification task.
            __path (str): Path where to save intermediate optimizations.

    """

    def __init__(self, model, **kwargs):
        available_kwargs = ['callbacks', 'mode', 'class_weight', 'path', 'batch_size']
        assert all([kwarg in available_kwargs for kwarg in kwargs.keys()])
        self.model = model
        self.best_model_name = None
        self.__callbacks = kwargs['callbacks'] if 'callbacks' in kwargs else None
        self.__class_weight = kwargs['class_weight'] if 'class_weight' in kwargs else None
        self.__path = kwargs['path'] if 'path' in kwargs else None

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def train(self, x_train, y_train, x_val=None, y_val=None, **kwargs):
        """ Model training.

            Parameters:
                x_train (list, np.array): Input data for training.
                y_train (list, np.array): Output data for training.
                x_val (list, np.array): Input data for validation.
                y_val (list, np.array): Output data for validation.
        """

        # Train model
        training_kwargs = kwargs.copy()
        training_kwargs.update({'x': x_train,
                                'y': y_train})
        if not any([val_ is None for val_ in [x_val, y_val]]):
            training_kwargs.update({'validation_data': (x_val, y_val)})
        self.model.fit(**training_kwargs)

        # Best model
        if self.best_model_name:
            best_model_files = glob.glob(self.best_model_name + '*')
            if len(best_model_files) > 0:
                self.model.load_weights(filepath=self.best_model_name)
                for filename in glob.glob(self.best_model_name + '*'):
                    os.remove(filename)

    def predict(self, x, **kwargs):
        """ Model prediction.

            Parameters:
                x (list, np.array): Input data for prediction.
        """
        return self.model.predict(x, **kwargs)
