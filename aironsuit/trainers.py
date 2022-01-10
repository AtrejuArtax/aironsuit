import glob
import os
import tensorflow as tf


class AIronTrainer(object):
    """ AIronTrainer is a module wrapper that takes care of the module training and prediction.

        Attributes:
            module (Module): NN module.
            best_module_name (str): Best module name.

    """

    def __init__(self, module):
        self.module = module
        self.best_module_name = None

    def __setattr__(self, key, value):
        self.__dict__[key] = value

    def fit(self, x_train, y_train, x_val=None, y_val=None, verbose=0, **kwargs):
        """ Module for fitting.

            Parameters:
                x_train (list, np.array): Input data for training.
                y_train (list, np.array): Output data for training.
                x_val (list, np.array): Input data for validation.
                y_val (list, np.array): Output data for validation.
                verbose (int): Level of verbosity.
        """
        fit(module=self.module, x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, verbose=verbose, **kwargs)

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


def fit(module, x_train, y_train=None, x_val=None, y_val=None, sample_weight=None, sample_weight_val=None,
        best_module_name=None, **kwargs):
    # ToDo: refactor this function
    # Train module
    training_kwargs = kwargs.copy()
    training_args = [x_train]
    if y_train is not None:
        training_args += [y_train]
    if sample_weight is not None:
        training_kwargs['sample_weight'] = sample_weight
    val_data = []
    for val_data_ in [x_val, y_val, sample_weight_val]:
        if val_data_ is not None:
            val_data += [val_data_]
    if len(val_data) != 0:
        training_kwargs.update({'validation_data': tuple(val_data)})
    if all([isinstance(data, tf.data.Dataset) for data in training_args]):
        # ToDo: make use of tfrecords for validation data too
        training_kwargs['validation_data'] = \
            tuple([tf.convert_to_tensor(list(val_data_.as_numpy_iterator()))
                   for val_data_ in training_kwargs['validation_data']])
        if sample_weight is not None:
            training_args += [training_kwargs['sample_weight']]
            del training_kwargs['sample_weight']
            training_args = tf.data.Dataset.from_tensor_slices(
                tuple([list(train_data_.as_numpy_iterator()) for train_data_ in training_args]))
        else:
            training_args = tf.data.Dataset.zip(tuple(training_args))
        training_args = training_args.batch(kwargs['batch_size'])
        module.fit(training_args, **training_kwargs)
    else:
        module.fit(*training_args, **training_kwargs)

    # Best module
    if best_module_name:
        best_module_files = glob.glob(best_module_name + '*')
        if len(best_module_files) > 0:
            module.load_weights(filepath=best_module_name)
            for filename in glob.glob(best_module_name + '*'):
                os.remove(filename)


def evaluate(module, x_val, y_val):
    return module.evaluate(x_val, y_val)
