from skorch import callbacks


def get_basic_callbacks(path=None, patience=3, model_name=None, verbose=0, epochs=None):
    basic_callbacks = []
    board_dir = path
    model_name_ = model_name if model_name else 'NN'
    board_dir += model_name_ + '_logs'
    basic_callbacks.append({'LRScheduler':
                                {'callback': callbacks.LRScheduler,
                                 'kwargs': dict(policy='CosineAnnealingLR',
                                                T_max=epochs - 1 if epochs else None)}})
    basic_callbacks.append({'EarlyStopping':
                                {'callback': callbacks.EarlyStopping,
                                 'kwargs': dict(
                                     monitor='val_loss',
                                     patience=patience,
                                     lower_is_better=True)}})
    basic_callbacks.append({'Checkpoint':
                                {'callback': callbacks.Checkpoint,
                                 'kwargs': dict(
                                     dirname=path + model_name_,
                                     save_best_only=True,
                                     save_weights_only=True)}})
    return basic_callbacks
