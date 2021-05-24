from tensorflow.keras import callbacks


def get_basic_callbacks(path, name, ext=None):
    callbacks = {}
    board_dir = path + name + '_logs'
    if ext:
        board_dir += '_' + str(ext)
    callbacks.update({'TensorBoard':
                          {'callback': callbacks.TensorBoard,
                           'kargs': dict(log_dir=board_dir)}})
    callbacks.update({'ReduceLROnPlateau':
                          {'callback': callbacks.ReduceLROnPlateau,
                           'kargs': dict(
                               monitor='val_loss',
                               factor=0.2,
                               patience=int(experiment_specs['early_stopping'] / 2),
                               min_lr=0.0000001,
                               verbose=verbose)}})
    callbacks.update({'EarlyStopping':
                          {'callback': callbacks.EarlyStopping,
                           'kargs': dict(
                               monitor='val_loss',
                               min_delta=0,
                               patience=experiment_specs['early_stopping'],
                               verbose=verbose,
                               mode='min')}})
    callbacks.update({'ModelCheckpoint':
                          {'callback': callbacks.ModelCheckpoint,
                           'kargs': dict(
                               filepath=path + name,
                               save_best_only=True,
                               save_weights_only=True,
                               verbose=verbose)}})
    return callbacks
