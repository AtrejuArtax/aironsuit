import os
import glob


def airon_trainer(x_train, y_train, x_val, y_val, model, train_specs, mode, path,
            verbose, batch_size, callbacks=None):

    best_model_name = path + 'best_epoch_model_' + mode

    # Callbacks
    callbacks_ = []
    if callbacks:
        for callback_dict in callbacks:
            callbacks_ += [callback_dict['callback'](callback_dict['kargs'])]
        best_model_files = glob.glob(best_model_name + '*')
        if len(best_model_files) > 0:
            for filename in glob.glob(best_model_name + '*'):
                os.remove(filename)

    # Train model
    class_weight = None if 'class_weight' not in train_specs.keys() \
        else {output_name: train_specs['class_weight'] for output_name in model.output_names}
    kargs = {'x': x_train,
             'y': y_train,
             'epochs': train_specs['epochs'],
             'callbacks': callbacks_,
             'class_weight': class_weight,
             'shuffle': True,
             'verbose': verbose,
             'batch_size': batch_size}
    if not any([val_ is None for val_ in [x_val, y_val]]):
        kargs.update({'validation_data': (x_val, y_val)})
    model.fit(**kargs)

    # Best model
    if callbacks:
        best_model_files = glob.glob(best_model_name + '*')
        if len(best_model_files) > 0:
            model.load_weights(filepath=best_model_name)
            for filename in glob.glob(best_model_name + '*'):
                os.remove(filename)