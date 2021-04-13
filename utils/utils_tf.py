from tensorflow.keras import models


def load_json(self, filepath):
    json_file = open(filepath, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    return models.model_from_json(loaded_model_json)