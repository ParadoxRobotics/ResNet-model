from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------

# normalization process
transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), # resize the image
     transforms.RandomHorizontalFlip(), # random flip
     transforms.ToTensor(), # image -> tensor
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose(
    [transforms.ToTensor(), # image -> tensor
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# load training dataset
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = data.DataLoader(train_data, batch_size = 5, shuffle = True)
print("Number of train samples: ", len(train_data))

# Load test dataset
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = data.DataLoader(test_data, batch_size = 5, shuffle = True)
print("Number of test samples: ", len(test_data))

#-------------------------------------------------------------------------------
#                  CNN classifier class definition
#-------------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        # create resiual block
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.SELU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False)
        )
        self.shortcut = shortcut

    # Block with residual connection
    def forward(self, x):
        out = self.block(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.selu(out)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # input layer
        self.first_processing = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.SELU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # ResNet34 layer [3,4,6,3]
        self.layer_1 = self.build_block(64, 128, 3)
        self.layer_2 = self.build_block(128, 256, 4, stride=2)
        self.layer_3 = self.build_block(256, 512, 6, stride=2)
        self.layer_4 = self.build_block(512, 512, 3, stride=2)

    def build_block(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
        )
        layers = []
        layers.append(ResBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.first_processing(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = F.avg_pool2d(out, out.size()[3])
        return out

# CNN based on the Donecle general classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Pretrained residual CNN
        # ResNet Image -> 64x64 + 3 channels RGB
        self.ResnetImage = ResNet()
        # Donecle classifier
        self.feedforward = nn.Sequential(OrderedDict([
            # linear 1
            ('lin1', nn.Linear(512, 128)),
            ('SELU1', nn.SELU()),
            ('drop1', nn.AlphaDropout(p=0.25)),
            # linear 2
            ('lin2', nn.Linear(128, 64)),
            ('SELU2', nn.SELU()),
            ('drop2', nn.AlphaDropout(p=0.25)),
            # linear 2
            ('lin3', nn.Linear(64, 32)),
            ('SELU3', nn.SELU()),
            ('drop3', nn.AlphaDropout(p=0.25)),
            # linear 2
            ('lin4', nn.Linear(32, 10)),
            ('soft1',nn.LogSoftmax()),
        ]))

    def forward(self, x):
        batch_size = x.size(0)
        # Process image and flat the output
        Res_out = self.ResnetImage(x).view(batch_size, -1)
        # feedforward
        pred = self.feedforward(Res_out)
        return pred

# Network instantiation
RN = Classifier()
print("ResNet50 STRUCTURE : \n", RN)

#-------------------------------------------------------------------------------
#                           Optimizer config
#-------------------------------------------------------------------------------

# SGD optimizer with Nesterow momentum
optimizer = optim.SGD(RN.parameters(), lr = 0.04,
                                            momentum = 0.90,
                                            weight_decay = 0.00001,
                                            nesterov = True)

# Training parameter
number_epoch = 20
# Learning rate scheduler (decreasing polynomial)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / number_epoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

#-------------------------------------------------------------------------------
#                            Learning procedure
#-------------------------------------------------------------------------------

def train(epoch):
    RN.train()
    i = 0 # to know the pass number between epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # training
        optimizer.zero_grad()
        output = RN(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print("step = ", i)
        i += 1
        # print loss over the datset
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

#-------------------------------------------------------------------------------
#                             test procedure
#-------------------------------------------------------------------------------

def test():
    with torch.no_grad():
        RN.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            # predict
            output = RN(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

#-------------------------------------------------------------------------------
#                             inference procedure
#-------------------------------------------------------------------------------

print("START TRAINING : \n")
for epoch in range(number_epoch):
    print("START TRAINING epoch <[O_O]> : \n", epoch)
    train(epoch)
    scheduler.step()
    print("\n\n START TESTING...please wait <[°_°]> : \n")
    test()
