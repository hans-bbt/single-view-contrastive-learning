'''contrast among patch and main image'''
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

        distances1 = (crop1 - main).pow(2).sum(1)  # squared distances
        distances2 = (crop2 - main).pow(2).sum(1)  # squared distances

        _, predicted1 = torch.max(crop1, dim=1)
        _, predicted2 = torch.max(crop2, dim=1)


        same1 = (predicted1 == labels)
        same2 = (predicted2 == labels)


        losses1 = 0.5 * (same1 * distances1 +
                        (1 + -1 * same1) * F.relu(self.margin - (distances1 + self.eps).sqrt()).pow(2))
        losses2 = 0.5 * (same2 * distances2 +
                        (1 + -1 * same2) * F.relu(self.margin - (distances2 + self.eps).sqrt()).pow(2))
        contraloss=0.1 * (losses1.mean() +  losses2.mean())

        CEloss=0.9*self.CEloss(main,labels)

        return contraloss+CEloss