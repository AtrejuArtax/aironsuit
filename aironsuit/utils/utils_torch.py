import torch


def save_model(model, name):
    torch.save(model, name)


def load_model(name, custom_objects=None):
    model = torch.load(name)
    model.eval()
    return model


def clear_session():
    pass


def summary(model):
    pass
