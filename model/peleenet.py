import torch
import torch.nn as nn
import torch.nn.functional as F


class PeleeNet(nn.Module):
    """
    PeleeNet
    See: https://arxiv.org/pdf/1804.06882.pdf for more details
    """
    def __init__(self, num_classes=1000, growth_rate=32, dense_layers=[3,4,8,6], bottleneck_widths=[1,2,4,4]):
        super(PeleeNet, self).__init__()
        stages = nn.Sequential()
        stages.add_module('stage_0', StemBlock())
        filters = 32
        for i in range(4):
            next_filters = filters + growth_rate * dense_layers[i]
            stages.add_module(f'stage_{i+1}', nn.Sequential(
                DenseBlock(filters, dense_layers[i], growth_rate, bottleneck_widths[i]),
                TransitionLayer(next_filters, next_filters, last=(i==3))))
            filters += growth_rate * dense_layers[i]
        self.stages = stages
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(704, num_classes)

    def forward(self, x):
        x = self.stages(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ConvBlock(nn.Module):
    """
    Conv - BN - ReLU
    """
    def __init__(self, in_planes, out_planes, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, bias=False, **kwargs),
                        nn.BatchNorm2d(out_planes),
                        nn.ReLU(inplace=True),
                    )

    def forward(self, x):
        return self.conv(x)


class StemBlock(nn.Module):
    """
    StemBlock used in PeleeNet
    According to Pelee paper, it is motivated by
    Inception-v4 Szegedy et al. (2017) and DSOD Shen et al. (2017)
    This is used before the first dense layer
    """

    def __init__(self, k=32):
        super(StemBlock, self).__init__()
        self.conv1 = ConvBlock(3, k, kernel_size=3, stride=2, padding=1)
        self.left_conv1 = ConvBlock(k, k//2, kernel_size=1, stride=1)
        self.left_conv2 = ConvBlock(k//2, k, kernel_size=3, stride=2, padding=1)
        self.right = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_last = ConvBlock(k*2, k, kernel_size=1, stride=1)

    def forward(self, x):
        """
        x: input image of shape [batch, 3, 224, 224]
        """
        x = self.conv1(x)                   # [batch, 32, 112, 112]
        left = self.left_conv1(x)           # [batch, 16, 112, 112]
        left = self.left_conv2(left)        # [batch, 32, 112, 112]
        right = self.right(x)               # [batch, 32, 112, 112]
        x = torch.cat([left, right], dim=1) # [batch, 64, 112, 112]
        x = self.conv_last(x)               # [batch, 32,  56,  56]
        return x

class DenseLayer(nn.Module):
    """
    Two-way dense layer suggested by the paper
    """
    def __init__(self, in_planes, growth_rate, bottleneck_width):
        """
        bottleneck_width is usally 1, 2, or 4
        """
        super(DenseLayer, self).__init__()
        left = [None] * 2
        right = [None] * 3

        inter_channel = bottleneck_width * growth_rate / 2
        inter_channel = bottleneck_width * growth_rate // 2  # will be k/2, k, 2k depending on bottleneck_width = 1,2,4

        # Left side
        left[0] = ConvBlock(in_planes, inter_channel, kernel_size=1, stride=1)
        left[1] = ConvBlock(inter_channel, growth_rate//2, kernel_size=3, stride=1, padding=1)
        self.left = left

        # Right side
        right[0] = ConvBlock(in_planes, inter_channel, kernel_size=1, stride=1)
        right[1] = ConvBlock(inter_channel, growth_rate//2, kernel_size=3, stride=1, padding=1)
        right[2] = ConvBlock(growth_rate//2, growth_rate//2, kernel_size=3, stride=1, padding=1)
        self.right = right

    def forward(self, x):
        left = x
        for block in self.left:
            left = block(left)
        right = x
        for block in self.right:
            right = block(right)
        x = torch.cat([x, left, right], dim=1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, in_planes, no_dense_layers, growth_rate, bottleneck_width):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer(in_planes+growth_rate*i, growth_rate, bottleneck_width) for i in range(no_dense_layers)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    def __init__(self, inp, oup, last=False):
        super(TransitionLayer, self).__init__()
        conv = ConvBlock(inp, oup, kernel_size=1, stride=1)
        if not last:
            self.layer = nn.Sequential(conv, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.layer = conv

    def forward(self, x):
        return self.layer(x)




x = torch.rand((10, 3, 224, 224))
model = PeleeNet()
y = model(x)
print(y.shape)
