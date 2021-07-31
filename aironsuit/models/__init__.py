from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.models.models_tf import *
else:
    from aironsuit.models.models_torch import *
