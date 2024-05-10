import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        layers = []
        for i in range(2):
            layers += [
                nn.ReflectionPad3d(1),
                nn.Conv3d(features, features, kernel_size=3),
                nn.InstanceNorm3d(features),
            ]
            if i==0:
                layers += [
                    nn.ReLU(True)
                ]
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return input + self.model(input)