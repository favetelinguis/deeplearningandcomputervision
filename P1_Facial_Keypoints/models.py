## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from inferno.extensions.layers.reshape import AsMatrix


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32,64, 3)
        self.bnconv2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bnconv4 = nn.BatchNorm2d(256)
        self.adaptive = nn.AdaptiveAvgPool2d(1)
        #self.fc1 = nn.Linear(43264, 1000)
        #self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(256, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        #m = nn.Hardtanh(-1, 1)
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.leaky_relu(self.bnconv2(self.conv2(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.conv3(x)), 2)
        x = AsMatrix()(self.adaptive(F.leaky_relu(self.bnconv4(self.conv4(x)))))
        ### Flatten
        #x = x.view(-1, self.num_flat_features(x))
        ### Classifier
        #x = F.dropout(F.leaky_relu(self.fc1(x)), p=.5, training=self.training)
        #x = F.dropout(F.leaky_relu(self.fc2(x)), p=.6, training=self.training)
        #x = self.adaptive(x)
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetOld(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv2 = nn.Conv2d(32,64, 3)
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.fc1 = nn.Linear(43264, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 136)
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        m = nn.Hardtanh(-1, 1)
        x = F.dropout(F.max_pool2d(F.elu(self.conv1(x)), 2), p=.1, training=self.training)
        x = F.dropout(F.max_pool2d(F.elu(self.conv2(x)), 2), p=.2, training=self.training)
        x = F.dropout(F.max_pool2d(F.elu(self.conv3(x)), 2), p=.3, training=self.training)
        x = F.dropout(F.max_pool2d(F.elu(self.conv4(x)), 2), p=.4, training=self.training)
        ### Flatten
        x = x.view(-1, self.num_flat_features(x))
        ### Classifier
        x = F.dropout(F.elu(self.fc1(x)), p=.5, training=self.training)
        x = F.dropout(F.elu(self.fc2(x)), p=.6, training=self.training)
        x = self.fc3(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
