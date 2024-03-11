import torch.nn as nn
from torchsummary import summary
from collections import OrderedDict


# 定义残差块
class Resnet_block(nn.Module):
    def __init__(self, in_channels):
        super(Resnet_block, self).__init__()
        block = []
        for i in range(2):
            block += [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_channels, in_channels, 3, 1, 0),
                      nn.InstanceNorm2d(in_channels),
                      nn.ReLU(True) if i > 0 else nn.Identity()]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = x + self.block(x)
        return out


class Cycle_Gan_G(nn.Module):
    def __init__(self):
        super(Cycle_Gan_G, self).__init__()
        net_dic = OrderedDict()
        # 三层卷积层
        net_dic.update({'first layer': nn.Sequential(
            nn.ReflectionPad2d(3),  # [3,256,256]  ->  [3,262,262]
            nn.Conv2d(3, 64, 7, 1),  # [3,262,262]  ->[64,256,256]
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )})
        net_dic.update({'second_conv': nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),  # [128,128,128]
            nn.InstanceNorm2d(128),
            nn.ReLU(True)
        )})
        net_dic.update({'three_conv': nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),  # [256,64,64]
            nn.InstanceNorm2d(256),
            nn.ReLU(True)
        )})

        # 9层 resnet block
        for i in range(6):
            net_dic.update({'Resnet_block{}'.format(i + 1): Resnet_block(256)})

        # up_sample
        net_dic.update({'up_sample1': nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),  # [128,128,128]
            nn.ReLU(True)
        )})
        net_dic.update({'up_sample2': nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),  # [64,256,256]
            nn.ReLU(True)
        )})

        net_dic.update({'last_layer': nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, 1),
            nn.Tanh()
        )})

        self.net_G = nn.Sequential(net_dic)
        self.init_weight()

    def init_weight(self):
        for w in self.modules():
            if isinstance(w, nn.Conv2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)

    def forward(self, x):
        out = self.net_G(x)
        return out


class Cycle_Gan_D(nn.Module):
    def __init__(self):
        super(Cycle_Gan_D, self).__init__()

        # 定义基本的卷积\bn\relu
        def base_Conv_bn_lkrl(in_channels, out_channels, stride):
            if in_channels == 3:
                bn = nn.Identity
            else:
                bn = nn.InstanceNorm2d
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 4, stride, 1),
                bn(out_channels),
                nn.LeakyReLU(0.2, True)
            )

        D_dic = OrderedDict()
        in_channels = 3
        out_channels = 64
        for i in range(4):
            if i < 3:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 2)})
            else:
                D_dic.update({'layer_{}'.format(i + 1): base_Conv_bn_lkrl(in_channels, out_channels, 1)})
            in_channels = out_channels
            out_channels *= 2
        D_dic.update({'last_layer': nn.Conv2d(512, 1, 4, 1, 1)})  # [batch,1,30,30]
        self.D_model = nn.Sequential(D_dic)

    def forward(self, x):
        return self.D_model(x)


if __name__ == '__main__':
    # G = Cycle_Gan_G().to('cuda')
    # summary(G, (3, 256, 256))
    D = Cycle_Gan_D().to('cuda')
    summary(D, (3, 256, 256))
