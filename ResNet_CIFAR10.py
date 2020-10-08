%%time
# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn, optim, cuda
from torchvision import datasets, transforms
from torch.autograd import Variable

# GPU settings
device = 'cuda' if cuda.is_available() else 'cpu'
print("Device: ", device, "\n");

# Training settings
batch_size = 256 

# CIFER10 Dataset
# train_dataset nomalize
train_dataset = datasets.CIFAR10(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

print("data shape:", train_dataset.data.shape)

train_mean = train_dataset.data.mean(axis=(0, 1, 2))
train_std = train_dataset.data.std(axis=(0, 1, 2))

train_mean = train_mean / 255
train_std = train_std / 255

print("train_mean:", train_mean)
print("train_std:", train_std)

# test_dataset nomalize
test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=transforms.ToTensor())

print("data shape:", test_dataset.data.shape)

test_mean = test_dataset.data.mean(axis=(0, 1, 2))
test_std = test_dataset.data.std(axis=(0, 1, 2))

test_mean = test_mean / 255
test_std = test_std / 255

print("test_mean:", test_mean)
print("test_std:", test_std)

# load
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(test_mean, test_std)
])

train_dataset = datasets.CIFAR10(root='./data',
                              train=True,
                              transform=train_transform,
                              download=True)
test_dataset = datasets.CIFAR10(root='./data/',
                              train=False,
                              transform=test_transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1) # /stride
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) # inplace 옵션의 역할이 무엇인가?
        self.conv2 =  nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # 
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.im = lambda x: x # identity mapping

    def forward(self, x):
        residual=self.im(x)
        if self.downsample is not None:
            residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out

class ResNet_CIFAR10(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False):
        super(ResNet_CIFAR10, self).__init__()
        self.in_channels = 64
        # 3, 32, 32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False) # bias?????????????????????
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # inplace???????????????
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 64, 32, 32
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 64, 32, 32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 128, 16, 16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 256, 8, 8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 512, 4, 4
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        # 512, 1, 1
        self.fc = nn.Linear(512, 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1: 
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),#??????????
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        for _ in range(1, blocks): 
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.mp(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.ap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.log_softmax(x)
        return x

model = ResNet_CIFAR10(ResidualBlock, [2, 2, 2, 2]) # ResNet20
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5) 

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = Variable(data), Variable(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        # data, target = Variable(data, volatile=True), Variable(target)
        data, target = data.to(device), target.to(device)
        output = model(data)
        # sum up batch loss
        test_loss += F.nll_loss(output, target, size_average=False).data
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('Test set({}): Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        epoch, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

accuracy=[]
for epoch in range(1, 100):
    train(epoch)
    accuracy.append(test(epoch))

%matplotlib inline
from matplotlib import pyplot as plt
plt.plot([i for i in range(len(accuracy))], accuracy)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Result')
plt.show()