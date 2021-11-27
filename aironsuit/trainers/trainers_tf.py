import os
import glob


def fit(module, x_train, y_train=None, x_val=None, y_val=None, sample_weight=None, sample_weight_val=None,
        best_module_name=None, **kwargs):

    # Train module
    training_kwargs = kwargs.copy()
    training_kwargs.update({'x': x_train})
    if y_train is not None:
        training_kwargs['y'] = y_train
    if sample_weight is not None:
        training_kwargs['sample_weight'] = sample_weight
    if not any([val_ is None for val_ in [x_val, y_val]]):
        if sample_weight_val is not None:
            training_kwargs.update({'validation_data': (x_val, y_val, sample_weight_val)})
        else:
            training_kwargs.update({'validation_data': (x_val, y_val)})
    module.fit(**training_kwargs)

    # Best module
    if best_module_name:
        best_module_files = glob.glob(best_module_name + '*')
        if len(best_module_files) > 0:
            module.load_weights(filepath=best_module_name)
            for filename in glob.glob(best_module_name + '*'):
                os.remove(filename)


def evaluate(module, x_val, y_val):
    return module.evaluate(x_val, y_val)
