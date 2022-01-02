import tensorflow as tf
from hyperopt.hp import uniform, choice
from tensorboard.plugins.hparams import api as hp


def choice_hp(name: str, values: list):
    hp_utils = dict(
        options=choice(name, values),
        logs=hp.HParam(name, hp.Discrete(values)))
    return hp_utils


def uniform_hp(name: str, min_value: float, max_value: float):
    hp_utils = dict(
        options=uniform(name, min_value, max_value),
        logs=hp.HParam(name, hp.RealInterval(min_value, max_value)))
    return hp_utils


def setup_design_logs(path, hyper_space, metric='val_loss'):
    with tf.summary.create_file_writer(path).as_default():
        hp.hparams_config(
            hparams=[value['logs'] for _, value in hyper_space.items()],
            metrics=[hp.Metric(metric, display_name=metric)],
        )


def update_design_logs(path, hparams, value, metric='val_loss', step=1):
    with tf.summary.create_file_writer(path).as_default():
        hp.hparams({value['logs']: key for key, value in hparams.items()})
        tf.summary.scalar(metric, value, step=step)
