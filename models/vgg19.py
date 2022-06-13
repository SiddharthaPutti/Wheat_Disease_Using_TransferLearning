import torch.nn as nn
from torch.nn import Sequential
import torchvision


class vgg19Pretrained(nn.Sequential):

    def __init__(self):
        super(vgg19Pretrained, self).__init__()

    def forward(self, model):
        model.avgpool = nn.AvgPool2d((5, 5))
        model.classifier = nn.Sequential(nn.Flatten(),
                                         nn.Linear(model.classifier.in_features, 512),
                                         )
        return model
