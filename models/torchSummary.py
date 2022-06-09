from models.att_unet_ori import AttUNet
from models.unet_ori import UNet
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(num_classes=26).to(device)

summary(model, input_size=(3,512,512))