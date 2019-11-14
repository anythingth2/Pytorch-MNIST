# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchsummary import summary
from torchvision.datasets import MNIST
from tqdm import tqdm, trange

# %%

mnist = MNIST('dataset',
              download=True,
              transform=transforms.Compose([
                  transforms.ToTensor()
              ])
              )

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
        x = F.softmax(x)

        return x

    def fit(self, data_loader, epochs=1,):
        log_period = 10
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        for ep in range(epochs):
            running_loss = 0.0
            results = []
            for minibatch_index, (x, y_true) in enumerate(tqdm(data_loader)):
                self.optimizer.zero_grad()
                y_pred = self.forward(x)
                loss = F.cross_entropy(y_pred, y_true)
                loss.backward()
                self.optimizer.step()

                results.extend(
                    list((torch.max(y_pred, 1)[1] == y_true).numpy()))
                running_loss += loss.item()
                if minibatch_index % log_period == 0:
                    tqdm.write(
                        f'loss: {running_loss / log_period} acc: {np.asarray(results).mean()}')
                    running_loss = 0.0


# %%
net = Net()
summary(net, (1, 28, 28))

# %%
train_loader = torch.utils.data.DataLoader(mnist,
                                           batch_size=32,
                                           shuffle=True,)
# %%

net.fit(train_loader, epochs=10)


# %%
