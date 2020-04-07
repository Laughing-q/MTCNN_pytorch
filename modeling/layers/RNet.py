import torch
from torch import nn


class RNet(nn.Module):
    def __init__(self, pretrained=False):
        super(RNet, self).__init__()
        # input 24x24
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)  # 22x22
        self.bn1 = nn.BatchNorm2d(28)
        self.prelu1 = nn.PReLU(28)
        self.maxpool1 = nn.MaxPool2d(3, 2, ceil_mode=True)  # 11x11
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(48)
        self.prelu2 = nn.PReLU(48)
        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)  # 4x4
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)  # 3x3
        self.bn3 = nn.BatchNorm2d(64)
        self.prelu3 = nn.PReLU(64)
        self.dense4 = nn.Linear(3 * 3 * 64, 128)
        self.prelu4 = nn.PReLU(128)
        self.dense5_1 = nn.Linear(128, 1)
        self.sigmoid5_1 = nn.Sigmoid()
        self.dense5_2 = nn.Linear(128, 4)

        # self.training = False
        #
        if pretrained:
            state_dict_path = 'weights/rnet.pth'
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
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        c = self.dense5_1(x)
        c = self.sigmoid5_1(c)
        b = self.dense5_2(x)
        return c, b


if __name__ == '__main__':
    a = torch.rand(128, 3, 24, 24)
    net = RNet()
    conf, reg = net(a)
    print(conf.shape)
    print(reg.shape)
