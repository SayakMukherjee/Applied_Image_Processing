#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the implementation of context encoder as mentioned by
# Pathak et al. in "Context Encoders: Feature Learning by Inpainting".
# Referred from: 
# 1. https://github.com/pathak22/context-encoder/blob/master/train.lua [Original]
# 2. https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/context_encoder/models.py [PyTorch re-implementation]
# ---------------------------------------------------------------------------

import torch.nn as nn

class ContextEncoder(nn.Module):
    """Implementation of the context encoder model
    """

    def __init__(self, channels: int = 3):
        super(ContextEncoder, self).__init__()

        #encoder
        self.enc_layer1 = nn.Sequential(nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
                                        nn.LeakyReLU(0.2))

        self.enc_layer2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.LeakyReLU(0.2))

        self.enc_layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.LeakyReLU(0.2))

        self.enc_layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.LeakyReLU(0.2))

        self.enc_layer5 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.LeakyReLU(0.2))

        self.channel_fc = nn.Conv2d(512, 4000, kernel_size=1)

        #decoder
        self.dec_layer1 = nn.Sequential(nn.ConvTranspose2d(4000, 512, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU())

        self.dec_layer2 = nn.Sequential(nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU())

        self.dec_layer3 = nn.Sequential(nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU())

        self.dec_layer4 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU())
                                        
        self.dec_layer5 = nn.Sequential(nn.ConvTranspose2d(64, channels, kernel_size=3, stride=1, padding=1),
                                        nn.Tanh())

    def forward(self, x):

        out = self.enc_layer1(x)
        out = self.enc_layer2(out)
        out = self.enc_layer3(out)
        out = self.enc_layer4(out)
        out = self.enc_layer5(out)
        out = self.channel_fc(out)

        out = self.dec_layer1(out)
        out = self.dec_layer2(out)
        out = self.dec_layer3(out)
        out = self.dec_layer4(out)
        out = self.dec_layer5(out)

        return out
