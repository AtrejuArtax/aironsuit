from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.callbacks.callbacks_tf import *
else:
    from aironsuit.callbacks.callbacks_torch import *
