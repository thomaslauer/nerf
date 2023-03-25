import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

import matplotlib.pyplot as plt


# dataset = ImageFolder(root='/mnt/e/images', transform=T.Compose([
#     T.Resize((64, 64)),
#     T.ToTensor(),
#     T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ]))

dataset = torchvision.datasets.CIFAR100(root='/mnt/e/cifar100', train=True, download=True, transform=T.Compose([
    T.Resize((64, 64)),
    T.ToTensor(),
    # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
]))

print(len(dataset))


batch_size=64
num_epochs=10

dataloader=DataLoader(dataset, batch_size)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # define a u-net shaped model which compresses from a 3x64x64 to a 1x128 hidden layer through conv2d layers
        self.conv1=nn.Conv2d(3, 8, 3, padding=1, stride=2)
        self.conv2=nn.Conv2d(8, 16, 3, padding=1, stride=2)
        self.conv3=nn.Conv2d(16, 32, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1, stride=2)

        self.upscale1 = nn.ConvTranspose2d(128, 64, 3, padding=1, stride=2)
        self.upscale2 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2)
        self.upscale3=nn.ConvTranspose2d(32, 16, 3, padding=1, stride=2)
        self.upscale4=nn.ConvTranspose2d(16, 8, 3, padding=1, stride=2)
        self.upscale5=nn.ConvTranspose2d(8, 3, 3, padding=1, stride=2)

    def forward(self, x):

        sizes=[]
        sizes.append(x.size())

        x=F.relu(self.conv1(x))
        sizes.append(x.size())
        x=F.relu(self.conv2(x))
        sizes.append(x.size())
        x=F.relu(self.conv3(x))
        sizes.append(x.size())
        x = F.relu(self.conv4(x))
        sizes.append(x.size())
        x = F.relu(self.conv5(x))

        x = F.relu(self.upscale1(x, output_size=sizes.pop()))
        x = F.relu(self.upscale2(x, output_size=sizes.pop()))
        x=F.relu(self.upscale3(x, output_size=sizes.pop()))
        x=F.relu(self.upscale4(x, output_size=sizes.pop()))
        x=F.relu(self.upscale5(x, output_size=sizes.pop()))

        return x


net=Network().to('cuda')

criterion=nn.MSELoss()

optimizer=torch.optim.Adam(net.parameters(), lr=0.001)


losses = []

for epoch in range(num_epochs):
    for batch_idx, (features, labels) in enumerate(dataloader):

        features=features.to('cuda')

        optimizer.zero_grad()
        # 1. forward
        output=net(features)
        loss=criterion(output, features)
        # 2. backward
        loss.backward()
        # 3. update
        optimizer.step()
        # 4. logging
        if (batch_idx % 10 == 0):
            losses.append(loss.item())


            plt.clf()
            plt.yscale('log')
            plt.plot(losses)
            plt.savefig("losses.png")

            print(
                f'epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss.item()}')

            # save input and output images
            save_image(features[0], 'input.png')
            save_image(output[0], 'output.png')
