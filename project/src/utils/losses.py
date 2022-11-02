#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the implementation of additional loss functions. 
# This is an extension and not implemented in the original paper. 
# It is adapted from assignement 3.
# ---------------------------------------------------------------------------

import torch

def content_loss(input_features, content_features, content_layers):
    """Calculates the content loss as in Gatys et al. 2016.

    Args:
        input_features (dict):  VGG features of the image to be optimized. It is a 
                                dictionary containing the layer names as keys and the corresponding 
                                features volumes as values.
        content_features (dict): VGG features of the content image. It is a dictionary 
                                 containing the layer names as keys and the corresponding features 
                                 volumes as values.
        content_layers (list): List containing which layers to consider for calculating
                               the content loss

    Returns:
        loss (Tensor): Content loss
    """

    loss = 0.

    for layer in content_layers:

        loss +=  ((input_features[layer] -
                 content_features[layer].detach()).pow(2)).mean()

    return loss / (len(content_layers))


def gram_matrix(x):
    """Calculates the gram matrix for a given feature matrix

    Args:
        x (Tensor): feature matrix of size (b, c, h, w) 

    Returns:
        x (Tensor): gram matrix
    """

    b, c, h, w = x.shape

    x = torch.bmm(x.view(b, c, -1), x.view(b, c, -1).transpose(1, 2))

    return x / (c*h*w)


def style_loss(input_features, style_features, style_layers):
    """Calculates the style loss as in Gatys et al. 2016.

    Args:
        input_features (dict):  VGG features of the image to be optimized. It is a 
                                dictionary containing the layer names as keys and the corresponding 
                                features volumes as values.
        style_features (dict):  VGG features of the style image. It is a dictionary 
                                containing the layer names as keys and the corresponding features 
                                volumes as values.
        style_layers (list):    a list containing which layers to consider for calculating
                                the style loss

    Returns:
        loss (Tensor): Style loss
    """

    loss = 0.

    for layer in style_layers:
        loss += ((gram_matrix(input_features[layer]) -
                 gram_matrix(style_features[layer].detach())).pow(2)).mean()

    return loss / (len(style_layers))