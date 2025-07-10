# from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
import torch
import torch.nn as nn
from torch.nn import functional as F

# register models
from ddfnet.ddf.ddf import DDFPack

from Res2Net.res2net import res2net101_26w_4s

def conv(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, isReLU=True):
    if isReLU:
        return nn.Sequential(
            nn.Conv2d(in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=((kernel_size - 1) * dilation) // 2,
                        bias=True), nn.ReLU())
    else:
        return nn.Sequential(
                nn.Conv2d(in_channel,
                        out_channel,
                        kernel_size=kernel_size,
                        stride=stride,
                        dilation=dilation,
                        padding=((kernel_size - 1) * dilation) // 2,
                        bias=True))

class WSPBlock(nn.Module):
    def __init__(self, channels):
        super(WSPBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x_weight = self.conv(x)
        x_weight = self.bn(x_weight)
        x_weight = self.relu(x_weight)
        out = torch.mul(x, x_weight)
        return out


# Global variance pooling
class GVP(nn.Module): 
    def __init__(self):
        super(GVP, self).__init__()
        self.mu_x_pool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.sig_x_pool  = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.refl = nn.ReflectionPad2d(1)

    def forward(self, x):
        x = self.refl(x)
        mu_x = self.mu_x_pool(x)
        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        out = sigma_x

        return out

class IQANet_DDF_Hyper(nn.Module):
    """
    Hyper network for learning perceptual rules.

    Args:
        hyper_in_channels: input feature channels for hyper network.
        feature_size: input feature map width/height for hyper network.
        target_in_size: input vector size for target network.
        target_fc(i)_size: fully connection layer size of target network.
        

    Note:
        For size match, input args must satisfy: 'target_fc(i)_size * target_fc(i+1)_size' is divisible by 'feature_size ^ 2'.
    """

    def __init__(self, hyper_in_channels, feature_size, target_fc1_size, target_fc2_size):
        super(IQANet_DDF_Hyper, self).__init__()

        self.hyperInChn = hyper_in_channels
        self.feature_size = feature_size
        self.f1 = target_fc1_size
        self.f2 = target_fc2_size
        self.target_in_size = 2048

        # model = models.resnet101(True)
        # self.block1 = nn.Sequential(*list(model.children())[:8])
        
        model = res2net101_26w_4s(pretrained=True)
        self.block1 = nn.Sequential(*list(model.children())[:7])
        self.block2 = nn.Sequential(*list(model.children())[7:8])

        # Conv layers for resnet output features
        channels = 2048
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, self.hyperInChn, 1, padding=(0, 0)),
            nn.ReLU(inplace=True),
            WSPBlock(self.hyperInChn),
            nn.ReLU(inplace=True),)
        
        self.ddf1 = DDFPack(channels, kernel_size=3, stride=1, dilation=1, head=1,
                se_ratio=0.2, nonlinearity='relu', kernel_combine='mul')
        
        self.ddf2 = DDFPack(channels, kernel_size=3, stride=1, dilation=1, head=1,
                se_ratio=0.4, nonlinearity='linear', kernel_combine='mul')
        
        self.ddf3 = DDFPack(channels, kernel_size=3, stride=1, dilation=1, head=1,
                se_ratio=0.5, nonlinearity='linear', kernel_combine='mul')
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (1,1))
        self.varpool = GVP()

        # Hyper network part, conv for generating target fc weights, fc for generating target fc biases
        self.fc1w_conv = nn.Conv2d(self.hyperInChn*2, int(self.target_in_size*2 * self.f1 / (feature_size *feature_size* 4 /3)), 3,  padding=(1, 1))
        self.fc1b_fc = nn.Linear(self.hyperInChn*2, self.f1)

        self.fc2w_conv = nn.Conv2d(self.hyperInChn*2, int(self.f1 * self.f2 / (feature_size *feature_size* 4 /3)), 3, padding=(1, 1))
        self.fc2b_fc = nn.Linear(self.hyperInChn*2, self.f2)

        self.fc3w_fc = nn.Linear(self.hyperInChn*2, self.f2)
        self.fc3b_fc = nn.Linear(self.hyperInChn*2, 1)

        # initialize
        for i, m_name in enumerate(self._modules):
            if i > 10:
                nn.init.kaiming_normal_(self._modules[m_name].weight.data)

        # freeze conv and weight of batchnorm
        for para in (self.block1.parameters()):
            para.requires_grad = False

        # freeze running mean and var of barchnorm
        self.block1.eval()

    def forward(self, x):

        feature_size = self.feature_size
        self.target_in_size = 2048

        res_out = self.block1(x)
        res_out = self.block2(res_out)
        
        res_out = self.ddf1(res_out)
        res_out = self.ddf2(res_out)
        res_out_new = self.ddf3(res_out)
        
        res_out_map = self.conv1(res_out)
    
        res_out1 = self.avgpool(res_out_new)
        res_out2 = self.varpool(res_out_new)

        target_in_vec = torch.cat([res_out1, res_out2],1) 

        res_out_map = F.interpolate(res_out_map, [feature_size, int(feature_size* 4 /3)], mode='bicubic', align_corners=False)
        hyper_in_feat = res_out_map.view(-1, self.hyperInChn, feature_size, int(feature_size* 4 /3))
        hyper_in_feat = torch.cat([hyper_in_feat, hyper_in_feat],1) 

        # generating target net weights & biases
        target_fc1w = self.fc1w_conv(hyper_in_feat).view(-1, self.f1, self.target_in_size*2, 1, 1)
        target_fc1b = self.fc1b_fc(self.avgpool(hyper_in_feat).squeeze()).view(-1, self.f1)

        target_fc2w = self.fc2w_conv(hyper_in_feat).view(-1, self.f2, self.f1, 1, 1)
        target_fc2b = self.fc2b_fc(self.avgpool(hyper_in_feat).squeeze()).view(-1, self.f2)

        target_fc3w = self.fc3w_fc(self.avgpool(hyper_in_feat).squeeze()).view(-1, 1, self.f2, 1, 1)
        target_fc3b = self.fc3b_fc(self.avgpool(hyper_in_feat).squeeze()).view(-1, 1)


        out = {}
        out['target_in_vec'] = target_in_vec
        out['target_fc1w'] = target_fc1w
        out['target_fc1b'] = target_fc1b
        out['target_fc2w'] = target_fc2w
        out['target_fc2b'] = target_fc2b
        out['target_fc3w'] = target_fc3w
        out['target_fc3b'] = target_fc3b

        return out


    def train(self, mode=True): 
        self.training = mode


        # for m in [self.resnet101, self.conv1, self.ddf1, self.conv2, self.ddf2, self.fc1w_conv, self.fc1b_fc, self.fc2w_conv, self.fc2b_fc, self.fc3w_fc, self.fc3b_fc]:
        # self.resnet101_freeze,

        for m in [self.block2, self.ddf1, self.ddf2, self.ddf3,  self.conv1,  self.fc1w_conv, self.fc1b_fc, self.fc2w_conv, self.fc2b_fc, self.fc3w_fc, self.fc3b_fc]:

            m.training = mode
            for module in m.children():
                module.train(mode)

        return self
    

class TargetNet(nn.Module):
    """
    Target network for quality prediction.
    """
    def __init__(self, paras):
        super(TargetNet, self).__init__()
        self.l1 = nn.Sequential(
            nn.Sigmoid(),  #
            TargetFC(paras['target_fc1w'], paras['target_fc1b']),
            nn.Sigmoid(),
        )
        self.l2 = nn.Sequential(
            TargetFC(paras['target_fc2w'], paras['target_fc2b']),
            nn.Sigmoid(),
        )

        self.l3 = TargetFC(paras['target_fc3w'], paras['target_fc3b'])


    def forward(self, x):
        q = self.l1(x)
        q = self.l2(q)
        q = self.l3(q).squeeze(2).squeeze(2)

        return q


class TargetFC(nn.Module):
    """
    Fully connection operations for target net

    Note:
        Weights & biases are different for different images in a batch,
        thus here we use group convolution for calculating images in a batch with individual weights & biases.
    """
    def __init__(self, weight, bias):
        super(TargetFC, self).__init__()
        self.weight = weight
        self.bias = bias

    def forward(self, input_):

        # input_re = input_
        input_re = input_.view(-1, input_.shape[0] * input_.shape[1], input_.shape[2], input_.shape[3])
        weight_re = self.weight.view(self.weight.shape[0] * self.weight.shape[1], self.weight.shape[2], self.weight.shape[3], self.weight.shape[4])
        bias_re = self.bias.view(self.bias.shape[0] * self.bias.shape[1])
        out = F.conv2d(input=input_re, weight=weight_re, bias=bias_re, groups=self.weight.shape[0])

        return out.view(input_.shape[0], self.weight.shape[1], input_.shape[2], input_.shape[3])
