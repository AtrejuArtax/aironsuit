import os


def get_backend():
    if 'AIRONSUIT_BACKEND' in os.environ:
        backend = os.environ['AIRONSUIT_BACKEND']
    else:
        backend = 'tensorflow'
    return backend
