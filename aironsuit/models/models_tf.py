from tensorflow.keras.models import Model


def get_latent_model(model, layer_names):
    return Model(inputs=model.inputs,
                 outputs=[layer.outputs for layer in model.layers
                          if any([layer_name in layer.name for layer_name in layer_names])])
