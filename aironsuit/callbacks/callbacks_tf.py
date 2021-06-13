from tensorflow.keras import callbacks


def get_basic_callbacks(path, early_stopping, model_name=None, ext=None, verbose=0):
    basic_callbacks = []
    board_dir = path
    model_name_ = model_name if model_name else 'NN'
    board_dir += model_name_ + '_logs'
    if ext:
        board_dir += '_' + str(ext)
    basic_callbacks.append({'TensorBoard':
                                {'callback': callbacks.TensorBoard,
                                 'kargs': dict(log_dir=board_dir)}})
    basic_callbacks.append({'ReduceLROnPlateau':
                                {'callback': callbacks.ReduceLROnPlateau,
                                 'kargs': dict(
                                     monitor='val_loss',
                                     factor=0.2,
                                     patience=int(early_stopping / 2),
                                     min_lr=0.0000001,
                                     verbose=verbose)}})
    basic_callbacks.append({'EarlyStopping':
                                {'callback': callbacks.EarlyStopping,
                                 'kargs': dict(
                                     monitor='val_loss',
                                     min_delta=0,
                                     patience=early_stopping,
                                     verbose=verbose,
                                     mode='min')}})
    basic_callbacks.append({'ModelCheckpoint':
                                {'callback': callbacks.ModelCheckpoint,
                                 'kargs': dict(
                                     filepath=path + model_name_,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     verbose=verbose)}})
    return basic_callbacks
