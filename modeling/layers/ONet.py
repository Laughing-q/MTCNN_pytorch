import torch
from torch import nn


class ONet(nn.Module):
    def __init__(self, pretrained=False):
        super(ONet, self).__init__()
        # input: 48x48
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)  # 46x46
        self.bn1 = nn.BatchNorm2d(32)
        self.prelu1 = nn.PReLU(32)
        self.maxpool1 = nn.MaxPool2d(3, 2, ceil_mode=True)  # 23x23
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)    # 21x21
        self.bn2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU(64)
        self.maxpool2 = nn.MaxPool2d(3, 2, ceil_mode=True)  # 10x10
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)       # 8x8
        self.bn3 = nn.BatchNorm2d(64)
        self.prelu3 = nn.PReLU(64)
        self.maxpool3 = nn.MaxPool2d(2, 2, ceil_mode=True)  # 4x4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)    # 3x3
        self.bn4 = nn.BatchNorm2d(128)
        self.prelu4 = nn.PReLU(128)
        self.dense5 = nn.Linear(3*3*128, 256)
        self.prelu5 = nn.PReLU(256)
        self.dense6_1 = nn.Linear(256, 1)
        self.sigmoid6_1 = nn.Sigmoid()
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        # self.training = False
        #
        if pretrained:
            state_dict_path = 'weights/onet.pth'
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
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.prelu4(x)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        c = self.dense6_1(x)
        c = self.sigmoid6_1(c)
        b = self.dense6_2(x)
        l = self.dense6_3(x)
        return c, b, l


if __name__ == '__main__':
    a = torch.rand(128, 3, 48, 48)
    net = ONet()
    conf, reg, mark = net(a)
    print(conf.shape)
    print(reg.shape)
    print(mark.shape)

