import torch
import torch.nn as nn


# class VGG(nn.Module):
#     # pooling layer at the front of block
#     def __init__(self, mode = 'rgb'):
#         super(VGG, self).__init__()
#
#         conv1 = nn.Sequential()
#         conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
#         conv1.add_module('bn1_1', nn.BatchNorm2d(64))
#         conv1.add_module('relu1_1', nn.ReLU(inplace=True))
#         conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
#         conv1.add_module('bn1_2', nn.BatchNorm2d(64))
#         conv1.add_module('relu1_2', nn.ReLU(inplace=True))
#         self.conv1 = conv1
#
#         conv2 = nn.Sequential()
#         conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
#         conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
#         conv2.add_module('bn2_1', nn.BatchNorm2d(128))
#         conv2.add_module('relu2_1', nn.ReLU())
#         conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
#         conv2.add_module('bn2_2', nn.BatchNorm2d(128))
#         conv2.add_module('relu2_2', nn.ReLU())
#         self.conv2 = conv2
#
#         conv3 = nn.Sequential()
#         conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
#         conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
#         conv3.add_module('bn3_1', nn.BatchNorm2d(256))
#         conv3.add_module('relu3_1', nn.ReLU())
#         conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
#         conv3.add_module('bn3_2', nn.BatchNorm2d(256))
#         conv3.add_module('relu3_2', nn.ReLU())
#         conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
#         conv3.add_module('bn3_3', nn.BatchNorm2d(256))
#         conv3.add_module('relu3_3', nn.ReLU())
#         self.conv3 = conv3
#
#         conv4 = nn.Sequential()
#         conv4.add_module('pool3_1', nn.MaxPool2d(2, stride=2))
#         conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
#         conv4.add_module('bn4_1', nn.BatchNorm2d(512))
#         conv4.add_module('relu4_1', nn.ReLU())
#         conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
#         conv4.add_module('bn4_2', nn.BatchNorm2d(512))
#         conv4.add_module('relu4_2', nn.ReLU())
#         conv4.add_module('conv4_3', nn.Conv2d(512, 512, 3, 1, 1))
#         conv4.add_module('bn4_3', nn.BatchNorm2d(512))
#         conv4.add_module('relu4_3', nn.ReLU())
#         self.conv4 = conv4
#
#         conv5 = nn.Sequential()
#         conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
#         conv5.add_module('conv5_1', nn.Conv2d(512, 512, 3, 1, 1))
#         conv5.add_module('bn5_1', nn.BatchNorm2d(512))
#         conv5.add_module('relu5_1', nn.ReLU())
#         conv5.add_module('conv5_2', nn.Conv2d(512, 512, 3, 1, 1))
#         conv5.add_module('bn5_2', nn.BatchNorm2d(512))
#         conv5.add_module('relu5_2', nn.ReLU())
#         conv5.add_module('conv5_3', nn.Conv2d(512, 512, 3, 1, 1))
#         conv5.add_module('bn5_3', nn.BatchNorm2d(512))
#         conv5.add_module('relu5_3', nn.ReLU())
#         self.conv5 = conv5
#
#         pre_train = torch.load('F:/WORKs_SDU/DPGN-main/pretrained/vgg16-397923af.pth')
#         self._initialize_weights(pre_train)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         return x
#
#     def _initialize_weights(self, pre_train):
#         # keys = pre_train.keys()
#         keys = list(pre_train.keys())
#
#         self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
#         self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
#         self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
#         self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
#         self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
#         self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
#         self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
#         self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
#         self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
#         self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
#         self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
#         self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
#         self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])
#
#         self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
#         self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
#         self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
#         self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
#         self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
#         self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
#         self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
#         self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
#         self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
#         self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
#         self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
#         self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
#         self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])

class VGG(nn.Module):
    # pooling layer at the front of block
    def __init__(self, mode = 'rgb'):
        super(VGG, self).__init__()

        conv1 = nn.Sequential()
        conv1.add_module('conv1_1', nn.Conv2d(3, 64, 3, 1, 1))
        conv1.add_module('bn1_1', nn.BatchNorm2d(64))
        conv1.add_module('relu1_1', nn.ReLU(inplace=True))
        conv1.add_module('conv1_2', nn.Conv2d(64, 64, 3, 1, 1))
        conv1.add_module('bn1_2', nn.BatchNorm2d(64))
        conv1.add_module('relu1_2', nn.ReLU(inplace=True))
        self.conv1 = conv1

        conv2 = nn.Sequential()
        conv2.add_module('pool1', nn.MaxPool2d(2, stride=2))
        conv2.add_module('conv2_1', nn.Conv2d(64, 128, 3, 1, 1))
        conv2.add_module('bn2_1', nn.BatchNorm2d(128))
        conv2.add_module('relu2_1', nn.ReLU())
        conv2.add_module('conv2_2', nn.Conv2d(128, 128, 3, 1, 1))
        conv2.add_module('bn2_2', nn.BatchNorm2d(128))
        conv2.add_module('relu2_2', nn.ReLU())
        self.conv2 = conv2

        conv3 = nn.Sequential()
        conv3.add_module('pool2', nn.MaxPool2d(2, stride=2))
        conv3.add_module('conv3_1', nn.Conv2d(128, 256, 3, 1, 1))
        conv3.add_module('bn3_1', nn.BatchNorm2d(256))
        conv3.add_module('relu3_1', nn.ReLU())
        conv3.add_module('conv3_2', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('bn3_2', nn.BatchNorm2d(256))
        conv3.add_module('relu3_2', nn.ReLU())
        conv3.add_module('conv3_3', nn.Conv2d(256, 256, 3, 1, 1))
        conv3.add_module('bn3_3', nn.BatchNorm2d(256))
        conv3.add_module('relu3_3', nn.ReLU())
        self.conv3 = conv3

        conv4 = nn.Sequential()
        conv4.add_module('pool3_1', nn.MaxPool2d(2, stride=2))
        conv4.add_module('conv4_1', nn.Conv2d(256, 512, 3, 1, 1))
        conv4.add_module('bn4_1', nn.BatchNorm2d(512))
        conv4.add_module('relu4_1', nn.ReLU())
        conv4.add_module('conv4_2', nn.Conv2d(512, 512, 3, 1, 1))
        conv4.add_module('bn4_2', nn.BatchNorm2d(512))
        conv4.add_module('relu4_2', nn.ReLU())
        conv4.add_module('conv4_33', nn.Conv2d(512, 256, 3, 1, 1))
        conv4.add_module('bn4_33', nn.BatchNorm2d(256))
        conv4.add_module('relu4_33', nn.ReLU())
        self.conv4 = conv4

        conv5 = nn.Sequential()
        conv5.add_module('pool4', nn.MaxPool2d(2, stride=2))
        conv5.add_module('conv5_11', nn.Conv2d(256, 256, 3, 1, 1))
        conv5.add_module('bn5_11', nn.BatchNorm2d(256))
        conv5.add_module('relu5_11', nn.ReLU())
        conv5.add_module('conv5_22', nn.Conv2d(256, 256, 3, 1, 1))
        conv5.add_module('bn5_22', nn.BatchNorm2d(256))
        conv5.add_module('relu5_22', nn.ReLU())
        conv5.add_module('conv5_33', nn.Conv2d(256, 256, 3, 1, 1))
        conv5.add_module('bn5_33', nn.BatchNorm2d(256))
        # conv5.add_module('bn5_2', nn.BatchNorm2d(256))
        conv5.add_module('relu5_33', nn.ReLU())
        self.conv5 = conv5

        pre_train = torch.load('./pretrained/vgg16-397923af.pth')
        self._initialize_weights(pre_train)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def _initialize_weights(self, pre_train):
        # keys = pre_train.keys()
        keys = list(pre_train.keys())

        self.conv1.conv1_1.weight.data.copy_(pre_train[keys[0]])
        self.conv1.conv1_2.weight.data.copy_(pre_train[keys[2]])
        self.conv2.conv2_1.weight.data.copy_(pre_train[keys[4]])
        self.conv2.conv2_2.weight.data.copy_(pre_train[keys[6]])
        self.conv3.conv3_1.weight.data.copy_(pre_train[keys[8]])
        self.conv3.conv3_2.weight.data.copy_(pre_train[keys[10]])
        self.conv3.conv3_3.weight.data.copy_(pre_train[keys[12]])
        self.conv4.conv4_1.weight.data.copy_(pre_train[keys[14]])
        self.conv4.conv4_2.weight.data.copy_(pre_train[keys[16]])
        # self.conv4.conv4_3.weight.data.copy_(pre_train[keys[18]])
        # self.conv5.conv5_1.weight.data.copy_(pre_train[keys[20]])
        # self.conv5.conv5_2.weight.data.copy_(pre_train[keys[22]])
        # self.conv5.conv5_3.weight.data.copy_(pre_train[keys[24]])

        self.conv1.conv1_1.bias.data.copy_(pre_train[keys[1]])
        self.conv1.conv1_2.bias.data.copy_(pre_train[keys[3]])
        self.conv2.conv2_1.bias.data.copy_(pre_train[keys[5]])
        self.conv2.conv2_2.bias.data.copy_(pre_train[keys[7]])
        self.conv3.conv3_1.bias.data.copy_(pre_train[keys[9]])
        self.conv3.conv3_2.bias.data.copy_(pre_train[keys[11]])
        self.conv3.conv3_3.bias.data.copy_(pre_train[keys[13]])
        self.conv4.conv4_1.bias.data.copy_(pre_train[keys[15]])
        self.conv4.conv4_2.bias.data.copy_(pre_train[keys[17]])
        # self.conv4.conv4_3.bias.data.copy_(pre_train[keys[19]])
        # self.conv5.conv5_1.bias.data.copy_(pre_train[keys[21]])
        # self.conv5.conv5_2.bias.data.copy_(pre_train[keys[23]])
        # self.conv5.conv5_3.bias.data.copy_(pre_train[keys[25]])


def vgg_XMZ(pretrained=False, **kwargs):
    """Constructs a VGG model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG()  # 去掉最后的全连接层fc的网络
    if pretrained:
        model_dict = model.state_dict()
        pretrained_dict = torch.load('F:/WORKs_SDU/DPGN-main/pretrained/vgg16-397923af.pth')  # 将原始VGG的全网络及参数加载进来
        # print(pretrained_dict.keys())
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)  # 将原始VGG中除fc外的所有层参数加载到model中
        model.load_state_dict(model_dict)
    return model