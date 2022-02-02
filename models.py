import torch
from torch import nn
from torchvision import transforms
import torch.nn.functional as F


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

