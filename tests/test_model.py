import pytest
import torch
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import pandas as pd

from typing import List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from challenge.challenge_p1 import SoftCrossEntropyLoss

def test_Net():
    """
    Tests the Net model developed in Challenge P2
    We input a single tesnor from test data, ( an image containing 7) and see if the mdoel is working fine!
    """
    # Load and Transform data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset2 = datasets.MNIST("../data", train=False, download = True, transform=transform)

    # define Model, optimizer and use checkpointed parameters
    model = Net()
    checkpoint = torch.load('./checkpoints/mnist_cnn.pt')
    model.load_state_dict(checkpoint['state_dict'])
    
    optimizer = optim.Adadelta(model.parameters(), lr=0.01)
    optimizer.load_state_dict(checkpoint['optimizer'])

    #Take a single image and pass it to the model
    
    single_loaded_img = dataset2.data[0]
    
    single_loaded_img = single_loaded_img.to('cpu')
    single_loaded_img = single_loaded_img[None, None]
    single_loaded_img = single_loaded_img.type('torch.FloatTensor') # instead of DoubleTensor

    # Prediction
    prediction_array = model(single_loaded_img)
    prediction = torch.argmax(prediction_array)
    
    assert torch.equal(prediction, torch.tensor(7))


class Net(nn.Module):
        def __init__(self):
            """Initialize layers for simple CNN arcitecture

            Returns
            ----------
            No return
            """
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.dropout1 = nn.Dropout2d(0.25)
            self.dropout2 = nn.Dropout2d(0.5)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass of CNN arcitecture

            Parameters
            ----------
                x : {torch.Tensor}
                    Input Tensor of shape (batch size, channels, height, width)

            Returns
            ----------
                prediction : torch.Tensor
                    Returns prediction tensor of shape (batch size, number of classes)
            """
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output





