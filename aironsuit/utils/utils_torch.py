import torch


def save_model(model, name):
    torch.save(model, name)


def load_model(name):
    model = torch.load(name)
    model.eval()
    return model


def clear_session():
    pass
