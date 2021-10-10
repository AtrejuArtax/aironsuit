# Databricks notebook source
import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
import torch.optim as optim
import torchvision
import os
import tempfile
os.environ['AIRONSUIT_BACKEND'] = 'pytorch'
from aironsuit.suit import AIronSuit

# COMMAND ----------

# Example Set-Up #

project_name = 'simplest_mnist'
num_classes = 10
input_shape = (28, 28, 1)
batch_size = 128
epochs = 10

# COMMAND ----------

# Load data
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(tempfile.gettempdir(), train=True, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST(tempfile.gettempdir(), train=False, download=True,
                             transform=torchvision.transforms.Compose([
                                 torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
  batch_size=32, shuffle=True)
x_train, y_train = train_loader.dataset.data, train_loader.dataset.targets
x_test, y_test = train_loader.dataset.data, train_loader.dataset.targets

# Preprocess data
x_train = np.expand_dims(np.array(x_train).astype('float32') / 255, -1)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[-1], x_train.shape[1], x_train.shape[2]))
x_test = np.expand_dims(np.array(x_test).astype('float32') / 255, -1)
y_train = np.array(y_train).reshape((len(y_train),))
y_test = np.array(y_train).reshape((len(y_test),))

# COMMAND ----------

# Create model


class LitMNIST(LightningModule):
    def __init__(self):
        super().__init__()

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        return x

    def training_step(self, batch, batch_idx=None):
        return {'loss': self._compute_loss(batch, batch_idx)}

    def validation_step(self, batch, batch_idx=None):
        return {'val_loss': self._compute_loss(batch, batch_idx)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}

    def _compute_loss(self, batch, batch_idx=None):
        x, y = batch
        y = y.type(torch.LongTensor)
        loss_f = nn.CrossEntropyLoss()
        return loss_f(self(x), y)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)


model = LitMNIST()

# COMMAND ----------

# Invoke AIronSuit
aironsuit = AIronSuit(model=model)
aironsuit.summary()

# COMMAND ----------

# Training
aironsuit.train(
    epochs=epochs,
    x_train=x_train,
    y_train=y_train)

# COMMAND ----------

# Evaluate
score = aironsuit.evaluate(x_test, y_test, use_trainer=True)
print('Test loss:', score)

# COMMAND ----------

# Save Model
aironsuit.save_model(os.path.join(os.path.expanduser("~"), project_name + '_model'))
