#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the implementation of discriminator as mentioned by
# Pathak et al. in "Context Encoders: Feature Learning by Inpainting".
# Referred from: 
# 1. https://github.com/pathak22/context-encoder/blob/master/train.lua [Original]
# 2. https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/models.py [PyTorch re-implementation]
# ---------------------------------------------------------------------------

import torch.nn as nn

class Discriminator(nn.Module):
    """Implementation of the discriminator model for adversarial loss
    """

    def __init__(self, channels: int = 3):
        super(Discriminator, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=3, stride=2, padding=1),
                                    nn.LeakyReLU(0.2))
                                    
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(128),
                                    nn.LeakyReLU(0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.2))

        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(512),
                                    nn.LeakyReLU(0.2))

        self.layer5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())

    def forward(self, x):

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out