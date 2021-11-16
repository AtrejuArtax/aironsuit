import os
import tempfile
from skorch import callbacks


def get_basic_callbacks(path=tempfile.gettempdir(), patience=3, name=None, verbose=0, epochs=None):
    basic_callbacks = []
    name = name if name is not None else 'NN'
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
                                     dirname=os.path.join(path, name))}})
    return basic_callbacks
