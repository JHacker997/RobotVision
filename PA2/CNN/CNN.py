__author__ = "John Hacker"

import torch.nn as nn


class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        self.input_layer = nn.Linear(input_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        features = self.input_layer(x)
        features = self.sigmoid(features)
        features = self.output_layer(features)

        # Uncomment the following return stmt once method implementation is done.
        return  features

class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        self.in_dim = 7 * 7 * 40
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.Sigmoid(),
            nn.MaxPool2d(2, stride=2))

        self.input_layer = nn.Linear(self.in_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        xSize = x.size(0)
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.view(xSize, -1)
        features = self.input_layer(features)
        features = self.sigmoid(features)
        features = self.output_layer(features)

        # Uncomment the following return stmt once method implementation is done.
        return  features


class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        self.in_dim = 7 * 7 * 40
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        

        self.input_layer = nn.Linear(self.in_dim, hidden_size)
        self.sigmoid = nn.Sigmoid()

        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        
        xSize = x.size(0)
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.view(xSize, -1)
        features = self.input_layer(features)
        features = self.sigmoid(features)
        features = self.output_layer(features)

        return features


class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        self.in_dim = 7 * 7 * 40
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))

        self.input_layer = nn.Linear(self.in_dim, hidden_size)
        self.middle_layer = nn.Linear(hidden_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        xSize = x.size(0)
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.view(xSize, -1)
        features = self.input_layer(features)
        features = self.sigmoid(features)
        features = self.middle_layer(features)
        features = self.sigmoid(features)
        features = self.output_layer(features)

        return features

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        self.in_dim = 7 * 7 * 40
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(40, 40, 5, stride=1, padding=2),
            nn.BatchNorm2d(40),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2))
        
        self.drop = nn.Dropout()
        self.input_layer = nn.Linear(self.in_dim, hidden_size)
        self.middle_layer = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features   #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        
        xSize = x.size(0)
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.view(xSize, -1)
        features = self.drop(features)
        features = self.input_layer(features)
        features = self.relu(features)
        features = self.middle_layer(features)
        features = self.sigmoid(features)
        features = self.output_layer(features)

        return features


class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size = args.hidden_size
        # model 1: base line
        if mode == 1:
            in_dim = 28*28 # input image size is 28x28
            self.model = Model_1(in_dim, self.hidden_size)

        # model 2: use two convolutional layer
        if mode == 2:
            self.model = Model_2(self.hidden_size)

        # model 3: replace sigmoid with relu
        if mode == 3:
            self.model = Model_3(self.hidden_size)

        # model 4: add one extra fully connected layer
        if mode == 4:
            self.model = Model_4(self.hidden_size)

        # model 5: utilize dropout
        if mode == 5:
            self.model = Model_5(self.hidden_size)


    def forward(self, x):
        if self.mode == 1:
            x = x.view(-1, 28 * 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
        # ======================================================================
        # Define softmax layer, use the features.
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign calculated value to logits after code implementation.
        logits = nn.functional.softmax(x, dim=1)
        return logits

