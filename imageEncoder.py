import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import torchvision.transforms as T
from torchvision.datasets import ImageFolder


dataset = ImageFolder(root='./images', transform=T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

print(len(dataset))


dataloader = DataLoader(dataset, batch_size=4)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # define a u-net shaped model which compresses from a 3x64x64 to a 1x128 hidden layer through conv2d layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))

        return x


net = Network()


features, labels = next(iter(dataloader))

print(features.shape)
print(labels)

print(net.forward(features).shape)


"""
batch_size = 8
num_epochs = 10

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)


for epoch in range(num_epochs):
    for batch_idx, (features, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        # 1. forward
        output = net(features)
        loss = criterion(output, labels)
        # 2. backward
        loss.backward()
        # 3. update
        optimizer.step()
        # 4. logging
        print(f'epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}')

"""
