import torch
import torch.nn as nn
import torch.nn.functional as F


class PeleeNet(nn.Module):
    """
    PeleeNet
    See: https://arxiv.org/pdf/1804.06882.pdf for more details
    """

    def __init__(self, num_classes=1000):
        stages = nn.Sequaltial()
        stages.add_module('stage_0', StemBlock())
        stages.add_module('stage_1', nn.Sequential(DenseBlock(3), TransitionLayer()))
        stages.add_module('stage_2', nn.Sequential(DenseBlock(4), TransitionLayer()))
        stages.add_module('stage_3', nn.Sequential(DenseBlock(8), TransitionLayer()))
        stages.add_module('stage_4', nn.Sequential(DenseBlock(6), TransitionLayer(last=True)))
        self.stages = stages
        self.pool = nn.AvgPool2d(kernel_size=7)
        self.classifier = nn.Linear(704, num_classes)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        return self.classifier(x)


class ConvBlock(nn.Module):
    """
    BN - Conv - RelU
    """
    def __init__(self, in_planes, out_planes, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
                        nn.Conv2d(in_planes, out_planes, **kwargs),
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
    def __init__(self, k):
        super(DenseLayer, self).__init__()
        left = [None] * 2
        right = [None] * 3

        # Left side
        left[0] = ConvBlock(k, 2*k, kernel_size=1, stride=1)
        left[1] = ConvBlock(2*k, k//2, kernel_size=3, stride=1, padding=1)
        self.left = left

        # Right side
        right[0] = ConvBlock(k, 2*k, kernel_size=1, stride=1)
        right[1] = ConvBlock(2*k, k//2, kernel_size=3, stride=1, padding=1)
        right[2] = ConvBlock(k//2, k//2, kernel_size=3, stride=1, padding=1)
        self.right = right

    def forward(self, x):
        left = x
        for block in self.left:
            left = block(left)
        right = x
        for block in self.right:
            right = block(right)
        x = torch.cat([left, x, right], dim=1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, no_dense_layers):
        super(DenseBlock, self).__init__()
        layers = [DenseLayer() for _ in range(no_dense_layers)]
        #FIXME


    def forward(self, x):
        #TODO
        pass


class TransitionLayer(nn.Module):
    def __init__(self, inp, oup, last=False):
        super(TransitionLayer, self).__init__()
        conv = ConvBlock(inp, oup, kernel_size=1, stride=1, padding=1)
        if not last:
            self.layer = nn.Sequential(conv, nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            self.layer = conv

    def forward(self, x):
        return self.layer(x)




x = torch.rand((10, 3, 224, 224))
model = PeleeNet()
model(x)
