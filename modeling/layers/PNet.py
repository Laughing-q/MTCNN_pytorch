import torch
from torch import nn


class PNet(nn.Module):
    def __init__(self, pretrained=False):
        super(PNet, self).__init__()  # input 12x12
        self.conv1 = nn.Conv2d(3, 10, 3, stride=1, padding=0)  # 10
        self.bn1 = nn.BatchNorm2d(10)
        self.prelu1 = nn.PReLU(10)
        self.maxpool1 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 5
        self.conv2 = nn.Conv2d(10, 16, 3, stride=1, padding=0)  # 3
        self.bn2 = nn.BatchNorm2d(16)
        self.prelu2 = nn.PReLU(16)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=1, padding=0)  # 1
        self.bn3 = nn.BatchNorm2d(32)
        self.prelu3 = nn.PReLU(32)
        self.conv4_1 = nn.Conv2d(32, 1, 1)  # 1
        self.sigmoid4_1 = nn.Sigmoid()
        self.conv4_2 = nn.Conv2d(32, 4, 1)  # 1

        # self.training = False

        if pretrained:
            state_dict_path = 'weights/pnet.pth'
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        c = self.conv4_1(x)
        c = self.sigmoid4_1(c)
        b = self.conv4_2(x)
        return c, b


if __name__ == '__main__':
    a = torch.rand(128, 3, 60, 60)
    net = PNet()
    conf, reg = net(a)
    print(conf.shape)
    print(reg.shape)
