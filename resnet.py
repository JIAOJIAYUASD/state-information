import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.utils.model_zoo as model_zoo
from torch.nn.utils import spectral_norm
#搭建resnet
#已阅

#后面会通过 model.load_state_dict(model_zoo.load_url(model_urls['resnet18'])) 来加载模型的参数
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
#下载路径，这里下载，仅仅是下载模型参数

class Bottleneck(nn.Module):
    expansion = 4
    #这里并没有做池化操作
    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)#二维卷积，输入的通道数，输出的通道数，卷积核的个数，是否使用偏置
        self.bn1 = nn.BatchNorm2d(planes)#Batch Normalization强行将数据拉回到均值为0，方差为1的正态分布上
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.is_last = is_last

    def forward(self, x):#无进行池化操作
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:#x是否需要下采样由x与out是否大小一样决定
            residual = self.downsample(x)#这个像素点就是窗口内所有像素的均值 Pk = (∑ Xi)/ S^2，有点像平均池化

        out += residual
        if not self.is_last:
            out = self.relu(out)#out = relu(P+ X下采样)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 输入img-> conv1 -> BatchNorm -> relu -> maxpool 2x2窗口的最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)# 2x2窗口的最大池化
        self.layer1 = self._make_layer(block, 64, layers[0])#通过layer这个list 记录层数
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, is_last=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes, bias=False)

        for m in self.modules():#遍历网络中的每一层，进行初始化操作
            if isinstance(m, nn.Conv2d):#判断一个变量是否是某个类型
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))# m.weight.data是卷积核参数,normal，正态分布赋值
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        #将卷积层封装 conv1 -> BatchNorm
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))#第一块
        self.inplanes = planes * block.expansion#记录输出输入的通道？？？
        for i in range(1, blocks-1):#2到n-1块
            layers.append(block(self.inplanes, planes))
        layers.append(block(self.inplanes, planes, is_last=is_last))#最后 一块

        return nn.Sequential(*layers)#返回一个通过list记录每一层数目的 CNN块

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feature_maps = self.layer4(x)

        x = self.avgpool(feature_maps)#平均池化
        x = x.view(x.size(0), -1)#batchsize，多维度的tensor展平成一维，其中-1表示会自适应的调整剩余的维度
        feature = x.renorm(2, 0, 1e-5).mul(1e5)#在第0维度对feat进行L2范数操作得到归一化结果。
        # 1e-5是代表maxnorm ，将大于1e-5的乘以1e5，使得最终归一化到0到1之间。
        w = self.fc.weight#记录fc层的权重
        ww = w.renorm(2, 0, 1e-5).mul(1e5)#归一化权值
        sim = feature.mm(ww.t())#这里就是计算内积

        return feature, sim, feature_maps#返回特征，特征图，内积图


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, residual_transform=None, output_activation='relu', norm='batch'):
        super(ResNetBasicblock, self).__init__()
        self.norm = norm

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'batch':
            self.bn_a = nn.BatchNorm2d(planes)#对输入的batch做归一化
        elif norm == 'instance':
            self.bn_a = nn.InstanceNorm2d(planes)#在图像像素上，对HW做归一化，用在风格化迁移；
        else:
            assert False, 'norm must be batch or instance'

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'batch':
            self.bn_b = nn.BatchNorm2d(planes)
        elif norm == 'instance':
            self.bn_b = nn.InstanceNorm2d(planes)
        else:
            assert False, 'norm must be batch or instance'

        self.residual_transform = residual_transform#残差做转化
        self.output_activation = nn.ReLU() if output_activation == 'relu' else nn.Tanh()#两个激活函数

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)
        # basicblock = F.leaky_relu(basicblock, 0.1, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)#归一化

        if self.residual_transform is not None:
            residual = self.residual_transform(x)

        if residual.size()[1] > basicblock.size()[1]:
            residual = residual[:, :basicblock.size()[1], :, :]#长度调整
        output = self.output_activation(residual + basicblock)#这里为什么要两个加起来？？？
        return output


def init_params(m):
    """
    initialize a module's parameters
    if conv2d or convT2d, using he normalization
    if bn set weight to 1 and bias to 0
    :param m:
    :return:
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))#下载resnet50，并作参数恢复
    return model

