from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.utils.utils_tf import *
else:
    from aironsuit.utils.utils_torch import *
