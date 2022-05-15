'''
Author: Peng Bo
Date: 2022-05-14 16:09:11
LastEditTime: 2022-05-15 20:53:50
Description: count the parameters for a given network 

'''

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

if __name__ == "__main__":

    from unet_dw import UNetDW
    from unet_ori import UNet
    from gln import GLN
    from gln2 import GLN2
    from globalNet import GlobalNet

    in_channels = [1, 1, 1,1]
    out_channels = [37, 19, 85,6]
    globalNet_params = {'scale_factor': 0.25,
                        'kernel_size': 3,
                        'dilations': [1, 2, 5, 2, 1]
                        }

    localNet_params = {'in_channels': in_channels,
                       'out_channels': out_channels}

    print('UNetDW', get_parameter_number(UNetDW(in_channels, out_channels)))
    print('UNet', get_parameter_number(UNet(in_channels, out_channels)))
    print('glonet', get_parameter_number(GlobalNet(in_channels, out_channels)))
    print('gu2net', get_parameter_number(
        GLN(UNetDW, localNet_params, globalNet_params)))
    print('g2u2net', get_parameter_number(
        GLN2(UNetDW, localNet_params, globalNet_params)))
    # model = GTN(u2net, localNet_params, gtn_params)
    import torch
    img = torch.zeros((4,1,512,512))
    # model(img)
