# # Databricks notebook source
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as f
# import torch.optim as optim
# import torchvision
# import os
# os.environ['AIRONSUIT_BACKEND'] = 'pytorch'
# from aironsuit.suit import AIronSuit
#
# # COMMAND ----------
#
# # Example Set-Up #
#
# project_name = 'simplest_mnist'
# num_classes = 10
# input_shape = (28, 28, 1)
# batch_size = 128
# epochs = 10
#
# # COMMAND ----------
#
# # Load data
# train_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                  torchvision.transforms.ToTensor(),
#                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
#   batch_size=32, shuffle=True)
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('/files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                  torchvision.transforms.ToTensor(),
#                                  torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
#   batch_size=32, shuffle=True)
# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
# # Preprocess data
# x_train = np.expand_dims(x_train.astype('float32') / 255, -1)
# x_test = np.expand_dims(x_test.astype('float32') / 255, -1)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
#
# # COMMAND ----------
#
# # Create model
#
#
# class Model(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = f.relu(f.max_pool2d(self.conv1(x), 2))
#         x = f.relu(f.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = f.relu(self.fc1(x))
#         x = f.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return f.log_softmax(x)
#
#
# model = Model()
# optimizer = optim.Adam(model.parameters(), lr=0.01)
#
# # COMMAND ----------
#
# # Invoke AIronSuit
# aironsuit = AIronSuit(model=model)
# aironsuit.summary()
#
# # COMMAND ----------
#
# # Training
# aironsuit.train(
#     epochs=epochs,
#     x_train=x_train,
#     y_train=y_train)
#
# # COMMAND ----------
#
# # Evaluate
# score = aironsuit.evaluate(x_test, y_test)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
#
# # COMMAND ----------
#
# # Save Model
# aironsuit.save_model(os.path.join(os.path.expanduser("~"), project_name + '_model'))
