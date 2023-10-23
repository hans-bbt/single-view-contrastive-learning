'''contrast among patch'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class loss_selfcontrast(nn.Module):
    def __init__(self ,margin=0.5):
        super(loss_selfcontrast, self).__init__()
        self.margin = margin
        self.eps = 1e-9
        self.CEloss=torch.nn.CrossEntropyLoss()

    def forward(self, main,crop1,crop2,labels, size_average=True):

        distances = (crop1 - crop2).pow(2).sum(1)  # squared distances
        _, predicted1 = torch.max(crop1, dim=1)
        _, predicted2 = torch.max(crop2, dim=1)
        same=(predicted1==predicted2)

        losses = 0.5 * (same * distances +
                        (1 + -1 * same) * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2)) #same=0/1

        CEloss=self.CEloss(main,labels)

        return 0.1*losses.mean()+0.9*CEloss