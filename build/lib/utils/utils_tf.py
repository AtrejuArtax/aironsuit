from tensorflow.keras import models
import tensorflow.keras.backend as K


def load_model(name):
    json_file = open(name + '_topology', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = models.model_from_json(loaded_model_json)
    model.load_weights(filepath=name + '_weights')
    return model


def save_model(model, name):
    model.save_weights(filepath=name + '_weights')
    with open(name + '_topology', "w") as json_file:
        json_file.write(model.to_json())


def clear_session():
    K.clear_session()
