import torch.nn as nn
from torchvision import models


class TripletNet(nn.Module):

    def __init__(self):
        super(TripletNet, self).__init__()
        model_ft = models.resnet18(pretrained=True)
        num_fc_inputs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_fc_inputs, 16)
        self.embedding_net = model_ft

    def forward(self, anchor, positive, negative):
        anchor_output = self.embedding_net(anchor)
        positive_output = self.embedding_net(positive)
        negative_output = self.embedding_net(negative)

        return anchor_output, positive_output, negative_output
