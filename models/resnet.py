import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo

BN = None

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet50c', 'resnet50d', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BN(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BN(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BN(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BN(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BN(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        bypass_bn_weight_list.append(self.bn3.weight)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_stem=False, 
                 avg_down=False, bypass_last_bn=False):
        
        global BN, bypass_bn_weight_list
        BN = nn.BatchNorm2d

        bypass_bn_weight_list = []


        self.inplanes = 64
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem
        self.avg_down = avg_down

        self.concat_up_512to128 = nn.ConvTranspose2d(512, 128, kernel_size=2, stride=2)
        self.concat_up_256to128 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.concat_conv_256to128 = nn.Conv2d(256, 128, kernel_size=1)
        self.concat_conv_128to128 = nn.Conv2d(128, 128, kernel_size=1)
        self.concat_conv_64to128 = nn.Conv2d(64, 128, kernel_size=1)

        self.out_conv_256to26 = nn.ConvTranspose2d(256, 26, kernel_size=2, stride=2)

        if self.deep_stem:
            self.conv1 = nn.Sequential(
                        nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
                        BN(32),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
                    )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BN(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def _make_layer(self, block, planes, blocks, stride=1, avg_down=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(stride, stride=stride, ceil_mode=True, count_include_pad=False),
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=1, bias=False),
                    BN(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    BN(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # suppose x - 3c,512*512
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)  # x_conv - 64c,256*256
        x_pool = self.maxpool(x_conv)   # x_pool - 64c,128*128

        layer1 = self.layer1(x_pool)    # layer1 - 64c,128*128
        layer2 = self.layer2(layer1)    # layer2 - 128c,64*64
        layer3 = self.layer3(layer2)    # layer3 - 256c,32*32
        layer4 = self.layer4(layer3)    # layer4 - 512c,16*16

        decoder1_1 = self.concat_up_512to128(layer4)        # decoder1_1 - 128c,32*32
        decoder1_1 = self.relu(decoder1_1)
        decoder1_2 = self.concat_conv_256to128(layer3)      # decoder1_2 - 128c,32*32
        decoder1_2 = self.relu(decoder1_2)
        decoder1 = torch.cat((decoder1_1, decoder1_2), 1)   # decoder1 - 256c,32*32

        decoder2_1 = self.concat_up_256to128(decoder1)      # decoder2_1 - 128c,64*64
        decoder2_1 = self.relu(decoder2_1)
        decoder2_2 = self.concat_conv_128to128(layer2)      # decoder2_2 - 128c,64*64
        decoder2_2 = self.relu(decoder2_2)
        decoder2 = torch.cat((decoder2_1, decoder2_2), 1)   # decoder2 - 256c,64*64

        decoder3_1 = self.concat_up_256to128(decoder2)      # decoder3_1 - 128c,128*128
        decoder3_1 = self.relu(decoder3_1)
        decoder3_2 = self.concat_conv_64to128(layer1)       # decoder3_2 - 128c,128*128
        decoder3_2 = self.relu(decoder3_2)
        decoder3 = torch.cat((decoder3_1, decoder3_2), 1)   # decoder3 - 256c,128*128

        decoder4_1 = self.concat_up_256to128(decoder3)      # decoder4_1 - 128c,256*256
        decoder4_1 = self.relu(decoder4_1)
        decoder4_2 = self.concat_conv_64to128(x_conv)       # decoder4_2 - 128c,256*256
        decoder4_2 = self.relu(decoder4_2)
        decoder4 = torch.cat((decoder4_1, decoder4_2), 1)   # decoder4 - 256c,256*256

        out = self.out_conv_256to26(decoder4)               # out - 26c,512*512
        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('/home/ubuntu/zhangjian/code/LandmarkDetection/experiments/template/resnet34-333f7ec4.pth'), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def resnet50c(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet50d(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], deep_stem=True, avg_down=True, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model
