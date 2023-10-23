import torch
import torch.nn as nn
import torchvision
from torchvision.models import AlexNet_Weights
import timm

class contrast_model(nn.Module):
    def __init__(self,  pretrained=False):
        super(contrast_model, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True, num_classes=6)
        self.model.load_state_dict(torch.load("store\\pth_\\Resnet18_finetuing_s224_n500.pth"))



    def forward(self, main,crop1=None,crop2=None,mode="test"):
        main=self.model(main)
        if mode=="train":
            crop1 = self.model(crop1)
            crop2 = self.model(crop2)

            return main,crop1,crop2

        return main