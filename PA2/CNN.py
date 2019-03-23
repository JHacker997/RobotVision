__author__ = "Aisha Urooj"

import torch.nn as nn


class Model_1(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super().__init__()
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(hidden_size, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

class Model_2(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()


class Model_3(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

class Model_4(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features    #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()

class Model_5(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Uncomment the following stmt with appropriate input dimensions once model's implementation is done.
        # self.output_layer = nn.Linear(in_dim, 10)

    def forward(self, x):
        # ======================================================================
        # Forward input x through your model to get features   #
        #
        # ----------------- YOUR CODE HERE ----------------------
        #

        # Uncomment the following return stmt once method implementation is done.
        # return  features
        # Delete line return NotImplementedError() once method is implemented.
        return NotImplementedError()


class Net(nn.Module):
    def __init__(self, mode, args):
        super().__init__()
        self.mode = mode
        self.hidden_size= args.hidden_size
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
            x = x.view(-1, 28* 28)
            x = self.model(x)
        if self.mode in [2, 3, 4, 5]:
            x = self.model(x)
        # ======================================================================
        # Define softmax layer, use the features.
        # ----------------- YOUR CODE HERE ----------------------
        #
        # Remove NotImplementedError and assign calculated value to logits after code implementation.
        logits = NotImplementedError
        return logits

