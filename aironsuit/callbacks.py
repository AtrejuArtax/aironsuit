import glob
import os
import tempfile

from tensorflow.keras import callbacks


def init_callbacks(raw_callbacks):
    callbacks_ = []
    if isinstance(raw_callbacks, list):
        for raw_callback in raw_callbacks:
            if isinstance(raw_callback, dict):
                callback_name = list(raw_callback.keys())[0]
                callback_ = raw_callback[callback_name]
                if 'Checkpoint' in callback_name:
                    path = callback_['kwargs']['dirname'] if 'dirname'in callback_['kwargs'].keys() \
                        else callback_['kwargs']['filepath']
                    best_model_name = os.path.join(path, 'best_epoch_model')
                    best_model_files = glob.glob(best_model_name + '*')
                    if len(best_model_files) > 0:
                        for filename in glob.glob(best_model_name + '*'):
                            os.remove(filename)
                    del best_model_files
                if 'kwargs' in callback_.keys():
                    callbacks_ += [callback_['callback'](**callback_['kwargs'])]
                else:
                    callbacks_ += [callback_['callback']()]
            else:
                callbacks_ += [raw_callback]
    return callbacks_


def get_basic_callbacks(path=tempfile.gettempdir(), patience=3, name=None, verbose=0, epochs=None, metric='val_loss',
                        mode='min'):
    basic_callbacks = []
    name = name if name else 'NN'
    basic_callbacks.append({'TensorBoard':
                                {'callback': callbacks.TensorBoard,
                                 'kwargs': dict(log_dir=os.path.join(path, name + '_logs'))}})
    basic_callbacks.append({'ReduceLROnPlateau':
                                {'callback': callbacks.ReduceLROnPlateau,
                                 'kwargs': dict(
                                     monitor=metric,
                                     factor=0.9,
                                     patience=int(patience / 2),
                                     min_lr=0.0000001,
                                     verbose=verbose,
                                     cooldown=1 + int(patience / 2),
                                     mode=mode)}})
    basic_callbacks.append({'EarlyStopping':
                                {'callback': callbacks.EarlyStopping,
                                 'kwargs': dict(
                                     monitor=metric,
                                     min_delta=0,
                                     patience=patience,
                                     verbose=verbose,
                                     mode=mode,
                                     restore_best_weights=True)}})
    return basic_callbacks
