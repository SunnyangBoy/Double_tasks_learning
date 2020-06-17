import torch
from torch import nn
import torch.nn.functional as F

'''
class DetectAngleModel(nn.Module):
    def __init__(self, resnet18):
        super(DetectAngleModel, self).__init__()
        self.resnet = nn.Sequential(*list(resnet18.children())[:-1])
        self.fc1 = nn.Linear(512, 2)
        self.fc2 = nn.Linear(512, 3)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.resnet(x)
        x = x.view(batch_size, -1)
        
        x1 = self.dropout1(x)
        x1 = self.fc1(x1)
        
        x2 = self.dropout2(x)
        x2 = self.fc2(x2)

        return x1, x2
'''


class DetectAngleModel(torch.nn.Module):
    def __init__(self):
        super(DetectAngleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(25088, 1024)

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 3)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = self.pool4(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        # 注意dropout不要重复使用一个
        # 注意分类问题的输出最好不要接relu，概率可以为负数，在softmax会考虑负概率的情况

        x1 = F.relu(self.fc2(x))
        x1 = self.dropout2(x1)
        x1 = self.fc3(x1)

        x2 = F.relu(self.fc4(x))
        x2 = self.dropout3(x2)
        x2 = self.fc5(x2)
        return x1, x2

