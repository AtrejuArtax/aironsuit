import os
import tempfile
from tensorflow.keras import callbacks


def get_basic_callbacks(path=tempfile.gettempdir(), patience=3, name=None, verbose=0, epochs=None):
    basic_callbacks = []
    name = name if name else 'NN'
    basic_callbacks.append({'TensorBoard':
                                {'callback': callbacks.TensorBoard,
                                 'kwargs': dict(log_dir=os.path.join(path, name + '_logs'))}})
    basic_callbacks.append({'ReduceLROnPlateau':
                                {'callback': callbacks.ReduceLROnPlateau,
                                 'kwargs': dict(
                                     monitor='val_loss',
                                     factor=0.2,
                                     patience=int(patience / 2),
                                     min_lr=0.0000001,
                                     verbose=verbose)}})
    basic_callbacks.append({'EarlyStopping':
                                {'callback': callbacks.EarlyStopping,
                                 'kwargs': dict(
                                     monitor='val_loss',
                                     min_delta=0,
                                     patience=patience,
                                     verbose=verbose,
                                     mode='min')}})
    basic_callbacks.append({'ModelCheckpoint':
                                {'callback': callbacks.ModelCheckpoint,
                                 'kwargs': dict(
                                     filepath=os.path.join(path, name),
                                     save_best_only=True,
                                     save_weights_only=True,
                                     verbose=verbose)}})
    return basic_callbacks
