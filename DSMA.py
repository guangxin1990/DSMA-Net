import torch
import torch.nn as nn
from torch.nn import functional as F


class Squeeze_Excite_Block(nn.Module):
    def __init__(self, channel, reduction):
        super(Squeeze_Excite_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.add_module('norm1', nn.BatchNorm2d(in_channels))
        self.add_module('relu1', nn.ReLU(True))
        self.add_module('conv1', nn.Conv2d(in_channels, 6*growth_rate, kernel_size=1,
                                          stride=1, bias=True))
        self.add_module('norm2', nn.BatchNorm2d(6*growth_rate))
        self.add_module('relu2', nn.ReLU(True))
        self.add_module('conv2', nn.Conv2d(6*growth_rate, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('SEblock', Squeeze_Excite_Block(channel=growth_rate, reduction=2))

    def forward(self, x):
        return super().forward(x)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(
            in_channels + i*growth_rate, growth_rate)
            for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_features = []

            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
                new_features.append(out)
            return torch.cat(new_features,1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], 1)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels, kernel_size=1,
                                          stride=1, padding=0, bias=True))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)


class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPBlock, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=3, dilation=1, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels[1], kernel_size=3, dilation=2, padding=2)

        out_channels = out_channels[0] + out_channels[1]
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        y = torch.cat([x1, x2], dim=1)
        y = self.conv_1x1(y)
        return y


class DMSA(nn.Module):
    def __init__(self, in_channels=3, down_blocks=(5, 5, 5), growth_rate=16, cur_channels=48, n_classes=2):
        super().__init__()

        self.down_blocks = down_blocks

        self.add_module('firstconv', nn.Conv2d(in_channels=in_channels, out_channels=cur_channels,
                                               kernel_size=3, stride=1, padding=1, bias=True))

        self.denseBlocksDown = nn.ModuleList([])
        self.transDownBlocks = nn.ModuleList([])
        for i in range(len(down_blocks)):
            self.denseBlocksDown.append(DenseBlock(cur_channels, growth_rate, down_blocks[i]))
            cur_channels += (growth_rate*down_blocks[i])
            self.transDownBlocks.append(TransitionDown(cur_channels))

        self.bn = nn.BatchNorm2d(cur_channels)
        self.linear = nn.Linear(cur_channels, n_classes)

        self.aspp = ASPPBlock(cur_channels, [int(cur_channels / 2), int(cur_channels / 2)])

        self.finalConv = nn.Conv2d(in_channels=cur_channels, out_channels=n_classes, kernel_size=1, stride=1,
                                   padding=0, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        out = self.firstconv(x)

        skip_connections = []
        for i in range(len(self.down_blocks)):
            out = self.denseBlocksDown[i](out)
            skip_connections.append(out)
            out = self.transDownBlocks[i](out)

        out_linear = F.adaptive_avg_pool2d(F.relu(self.bn(out)), 1)
        out_linear = out_linear.view(out_linear.size(0), -1)
        y = self.linear(out_linear)

        out = self.aspp(out)
        out = F.interpolate(out, size=224, mode='bilinear', align_corners=False)
        out = self.finalConv(out)
        out = self.softmax(out)
        return y, out


def DSMANet():
    return DMSA(in_channels=3, down_blocks=(6, 6, 6, 6, 6), growth_rate=16, cur_channels=64, n_classes=2)

