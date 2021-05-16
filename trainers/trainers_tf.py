import os
import glob


def airon_trainer(model, x_train, y_train, x_val, y_val, epochs, verbose=0, callbacks=None, **train_specs):

    class_weight = {output_name: train_specs['class_weight'] for output_name in model.output_names} \
        if 'class_weight' not in train_specs else None
    mode = train_specs['mode'] if 'mode' in train_specs else None
    path = train_specs['path'] if 'path' in train_specs else None
    batch_size = train_specs['batch_size'] if 'batch_size' in train_specs else 32

    best_model_name = None

    # Callbacks
    callbacks_ = []
    if callbacks:
        for callback_dict in callbacks:
            if callback_dict['name'] == 'ModelCheckpoint':
                ext = '_' + mode if mode else ''
                best_model_name = path + 'best_epoch_model' + ext
            callbacks_ += [callback_dict['callback'](callback_dict['kargs'])]
        best_model_files = glob.glob(best_model_name + '*')
        if len(best_model_files) > 0:
            for filename in glob.glob(best_model_name + '*'):
                os.remove(filename)

    # Train model
    kargs = {'x': x_train,
             'y': y_train,
             'epochs': epochs,
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
