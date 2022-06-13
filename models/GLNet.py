'''
Author: Peng Bo
Date: 2022-04-27 16:09:11
LastEditTime: 2022-05-21 18:42:06
Description: 

'''
import torch
import torch.nn as nn
from .unet_dw  import UNetDW
from .unet_ori  import UNet
from  .att_unet_ori import AttUNet
from .globalNet import globalNet


class GLNet(nn.Module):
    ''' global and local net '''
    def __init__(self, num_classes):
        super(GLNet, self).__init__()
        self.localNet = UNet(num_classes=num_classes)
        #self.localNet = UNetDW(num_classes=num_classes)
        self.globalNet = globalNet(3, num_classes)

    def forward(self, x):
        local_feature = self.localNet(x)
        fuse = torch.cat((x, local_feature), dim=1)
        global_feature = self.globalNet(fuse)
        return global_feature*local_feature
