from tensorflow.keras.models import model_from_json, Model
import tensorflow.keras.backend as bcknd
from inspect import getfullargspec


def load_model(name, custom_objects=None):
    json_file = open(name + '_topology', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json, custom_objects)
    model.load_weights(filepath=name + '_weights')
    return model


def save_model(model, name):
    model.save_weights(filepath=name + '_weights')
    with open(name + '_topology', "w") as json_file:
        json_file.write(model.to_json())


def clear_session():
    bcknd.clear_session()


def summary(model):
    """ Model summary.

        Parameters:
            model (Model): Model to summarize.
    """
    print('\n')
    print('________________________ Model Summary __________________________')
    print('Main model name: ' + model.name)
    print(model.summary())
    print('\n')
    print('_________________ Layers/Sub-Models Summaries ___________________')
    for layer in model.layers:
        print(layer.name)
        try:
            print(layer.summary())
        except:
            pass
