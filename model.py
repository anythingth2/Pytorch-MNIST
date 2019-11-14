# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchsummary import summary
import numpy as np
from tqdm import tqdm, trange
# %%
mnist = MNIST('dataset', download=True)

# %%


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3,)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3,)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2),)
        self.conv3 = nn.Conv2d(12, 6, kernel_size=3,)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=3,)
        self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2),)
        self.fc1 = nn.Linear(3*4*4, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = x.view(-1, 3*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x)
        return x

    def fit(self, x, y_true, epochs=10,):
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for ep in trange(epochs):
            y_pred = self.forward(x)
            loss = F.nll_loss(y_pred, y_true)
            tqdm.write(str(loss.item()))
            loss.backward()
            
            self.optimizer.step()
            


# %%
net = Net()
summary(net, (1, 28, 28))

# %%
x = mnist.train_data[:1000].unsqueeze(1).float()
y = mnist.train_labels[:1000]
# %%

net.fit(x, y)

# %%
