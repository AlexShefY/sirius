import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F

class M4(nn.Module):
    def __init__(self):
        super(M4, self).__init__()
        self.loader = 'M5()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 32, 4, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 4, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 4, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 4, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 4, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.conv6 = nn.Conv2d(160, 176, 4, bias=False)
        self.conv6_bn = nn.BatchNorm2d(176)
        self.conv7 = nn.Conv2d(176, 184, 4, bias=False)
        self.conv7_bn = nn.BatchNorm2d(184)
        self.fc1 = nn.Linear(9016, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        conv6 = F.relu(self.conv6_bn(self.conv6(conv5)))
        conv7 = F.relu(self.conv7_bn(self.conv7(conv6)))
        flat5 = torch.flatten(conv7.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)


class M5(nn.Module):
    def __init__(self):
        super(M5, self).__init__()
        self.loader = 'M5()'
        self.crop = transforms.CenterCrop(28)
        self.conv1 = nn.Conv2d(3, 32, 5, bias=False)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, bias=False)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 96, 5, bias=False)
        self.conv3_bn = nn.BatchNorm2d(96)
        self.conv4 = nn.Conv2d(96, 128, 5, bias=False)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 160, 5, bias=False)
        self.conv5_bn = nn.BatchNorm2d(160)
        self.fc1 = nn.Linear(10240, 10, bias=False)
        self.fc1_bn = nn.BatchNorm1d(10)
    def get_logits(self, x):
        x = self.crop((x - 0.5) * 2.0)
        conv1 = F.relu(self.conv1_bn(self.conv1(x)))
        conv2 = F.relu(self.conv2_bn(self.conv2(conv1)))
        conv3 = F.relu(self.conv3_bn(self.conv3(conv2)))
        conv4 = F.relu(self.conv4_bn(self.conv4(conv3)))
        conv5 = F.relu(self.conv5_bn(self.conv5(conv4)))
        flat5 = torch.flatten(conv5.permute(0, 2, 3, 1), 1)
        logits = self.fc1_bn(self.fc1(flat5))
        return logits
    def forward(self, x):
        return F.log_softmax(self.get_logits(x), dim=-1)

class denseNet(nn.Module):
  def __init__(self):
    from torchvision.models import densenet161
    super(denseNet, self).__init__()
    self.loader = 'denseNet()'
    self.net = densenet161()
    self.net.fc = nn.Linear(1024, 10)
  def get_logits(self, x):
    return self.net(x)
  def forward(self, x):
    return F.log_softmax(self.get_logits(x), dim=-1)

class resNet(nn.Module):
  def __init__(self):
    from torchvision.models import resnet18
    super(resNet, self).__init__()
    self.loader = 'resNet()'
    self.net = resnet18()
    self.net.fc = nn.Linear(512, 10)
  def get_logits(self, x):
    return self.net(x)
  def forward(self, x):
    return F.log_softmax(self.get_logits(x), dim=-1)

class CnnFnnModel(nn.Module):
    def __init__(self):
        super(CnnFnnModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 14 x 14
            nn.Dropout(p=0.25),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 7 x 7
            nn.Dropout(p=0.25),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 1024 x 3 x 3
            nn.Dropout(p=0.25),

            nn.Flatten(), 
            nn.Linear(1024*4*4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(p=0.25),
            nn.Linear(512, 10))
    def get_logits(self, x):
      return self.net(x)
   
    def forward(self, x):
      return F.log_softmax(self.get_logits(x), dim=-1)