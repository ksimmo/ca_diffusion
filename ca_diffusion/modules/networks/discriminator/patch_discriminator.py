import numpy as np

import torch
import torch.nn as nn

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class PatchDiscriminator2D(nn.Module):
    def __init__(self, channels_in, channels, n_layers=2, channels_max=512):
        super().__init__()

        layers = [nn.Conv2d(channels_in, channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for i in range(n_layers):
            channels_a = min(channels*int(np.power(2, i)), channels_max)
            channels_b = min(channels*int(np.power(2, i+1)), channels_max)

            layers += [nn.Conv2d(channels_a, channels_b, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.BatchNorm2d(channels_b),
                       nn.LeakyReLU(0.2, True)]
        channels_a = min(channels*int(np.power(2, n_layers)), channels_max)
        channels_b = min(channels*int(np.power(2, n_layers+1)), channels_max)
        layers += [nn.Conv2d(channels_a, channels_b, kernel_size=4, stride=1, padding=1, bias=False),
                       nn.BatchNorm2d(channels_b),
                       nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(channels_b, 1, kernel_size=4, stride=1, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
    

class PatchDiscriminator3D(nn.Module):
    def __init__(self, channels_in, channels, n_layers=2, channels_max=512):
        super().__init__()

        layers = [nn.Conv3d(channels_in, channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        for i in range(n_layers):
            channels_a = min(channels*int(np.power(2, i)), channels_max)
            channels_b = min(channels*int(np.power(2, i+1)), channels_max)

            layers += [nn.Conv3d(channels_a, channels_b, kernel_size=4, stride=2, padding=1, bias=False),
                       nn.BatchNorm3d(channels_b),
                       nn.LeakyReLU(0.2, True)]
        channels_a = min(channels*int(np.power(2, n_layers)), channels_max)
        channels_b = min(channels*int(np.power(2, n_layers+1)), channels_max)
        layers += [nn.Conv3d(channels_a, channels_b, kernel_size=4, stride=1, padding=1, bias=False),
                       nn.BatchNorm3d(channels_b),
                       nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv3d(channels_b, 1, kernel_size=4, stride=1, padding=1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

