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

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)

        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 128)
        self.linear3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))

        x = torch.flatten(x, 1)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


net = Network()


features, labels = next(iter(dataloader))

print(features.shape)
print(labels)

print(net.forward(features))


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

