import os
import glob
from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.callbacks.callbacks_tf import *
else:
    from aironsuit.callbacks.callbacks_torch import *


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
