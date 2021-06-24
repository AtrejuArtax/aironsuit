from aironsuit.backend import get_backend
BACKEND = get_backend()
if BACKEND == 'tensorflow':
    from aironsuit.trainers.trainers_tf import *
else:
    from aironsuit.trainers.trainers_torch import *
