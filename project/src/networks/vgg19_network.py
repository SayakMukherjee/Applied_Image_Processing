#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the implementation of pre-trained VGG19 network to 
# calculate the perceptual loss. This is an extension and not
# implemented in the original paper. It is adapted from assignement 3.
# ---------------------------------------------------------------------------

import torch.nn as nn
import torchvision.models as models

class Vgg19(nn.Module):

    def __init__(self, content_layers, style_layers, device):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features.to(device).eval()
        self.slices = []
        self.layer_names = []
        self._remaining_layers = set(content_layers + style_layers)
        self._conv_names = [
            'conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
            'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4',
            'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4',
        ]

        i = 0
        model = nn.Sequential()
        for layer in vgg.children():
            new_slice = False
            if isinstance(layer, nn.Conv2d):
                name = self._conv_names[i]
                i += 1

                if name in content_layers or name in style_layers:
                    new_slice = True
                    self.layer_names.append(name)
                    self._remaining_layers.remove(name)

            elif isinstance(layer, nn.ReLU):
                name = 'relu{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn{}'.format(i)

            model.add_module(name, layer)

            if new_slice:
                self.slices.append(model)
                model = nn.Sequential()
            
            if len(self._remaining_layers) < 1:
                break
        
        if len(self._remaining_layers) > 0:
            raise Exception('Not all layers provided in content_layes and/or style_layers exist.')

    def forward(self, x):
        outs = []
        for slice in self.slices:
            x = slice(x)
            outs.append(x.clone())

        out = dict(zip(self.layer_names, outs))
        return out