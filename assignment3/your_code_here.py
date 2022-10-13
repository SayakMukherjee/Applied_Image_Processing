import torch
import torch.optim as optim

import torchvision.transforms as transforms

from helper_functions import *


def normalize(img, mean, std):
    """ Normalizes an image tensor.

    # Parameters:
        @img, torch.tensor of size (b, c, h, w)
        @mean, torch.tensor of size (c)
        @std, torch.tensor of size (c)

    # Returns the normalized image
    """
    # TODO: 1. Implement normalization doing channel-wise z-score normalization.

    transform = transforms.Compose([transforms.Normalize(mean=mean, std=std)])

    return transform(img)


def content_loss(input_features, content_features, content_layers):
    """ Calculates the content loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @content_features, VGG features of the content image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @content_layers, a list containing which layers to consider for calculating
            the content loss.

    # Returns the content loss, a torch.tensor of size (1)
    """
    # TODO: 2. Implement the content loss given the input feature volume and the
    # content feature volume. Note that:
    # - Only the layers given in content_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.

    loss = 0.

    for layer in content_layers:
        loss += ((input_features[layer] -
                 content_features[layer].detach()).pow(2)).mean()

    return loss / (len(content_layers))


def gram_matrix(x):
    """ Calculates the gram matrix for a given feature matrix.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w) 

    # Returns the gram matrix
    """
    # TODO: 3.2 Implement the calculation of the normalized gram matrix.
    # Do not use for-loops, make use of Pytorch functionalities.

    b, c, h, w = x.shape

    x = torch.bmm(x.view(b, c, -1), x.view(b, c, -1).transpose(1, 2))

    return x / (c*h*w)


def style_loss(input_features, style_features, style_layers):
    """ Calculates the style loss as in Gatys et al. 2016.

    # Parameters:
        @input_features, VGG features of the image to be optimized. It is a 
            dictionary containing the layer names as keys and the corresponding 
            features volumes as values.
        @style_features, VGG features of the style image. It is a dictionary 
            containing the layer names as keys and the corresponding features 
            volumes as values.
        @style_layers, a list containing which layers to consider for calculating
            the style loss.

    # Returns the style loss, a torch.tensor of size (1)
    """
    # TODO: 3.1 Implement the style loss given the input feature volume and the
    # style feature volume. Note that:
    # - Only the layers given in style_layers should be used for calculating this loss.
    # - Normalize the loss by the number of layers.
    # - Implement the gram_matrix function.

    loss = 0.

    for layer in style_layers:
        loss += ((gram_matrix(input_features[layer]) -
                 gram_matrix(style_features[layer].detach())).pow(2)).mean()

    return loss / (len(style_layers))

    # return torch.tensor([0.], requires_grad=True)


def total_variation_loss(y):
    """ Calculates the total variation across the spatial dimensions.

    # Parameters:
        @x, torch.tensor of size (b, c, h, w)

    # Returns the total variation, a torch.tensor of size (1)
    """
    # TODO: 4. Implement the total variation loss.

    loss = torch.abs(torch.diff(y, dim=2)).sum() + \
        torch.abs(torch.diff(y, dim=3)).sum()

    return loss


def run_double_image(
        vgg_mean, vgg_std, content_img, style_img_1, style_img_2, num_steps,
        random_init, w_style_1, w_style_2, w_content, w_tv, content_layers, style_layers, device):

    # TODO: 5. Implement style transfer for two given style images.

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # 1. Normalize Input images
    normed_style_img_1 = normalize(style_img_1, vgg_mean, vgg_std)
    normed_style_img_2 = normalize(style_img_2, vgg_mean, vgg_std)
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    # Retrieve feature maps for content and style image
    style_features_1 = model(normed_style_img_1)
    style_features_2 = model(normed_style_img_2)
    content_features = model(normed_content_img)

    # Either initialize the image from random noise or from the content image
    if random_init:
        optim_img = torch.randn(content_img.data.size(), device=device)
        optim_img = torch.nn.Parameter(optim_img, requires_grad=True)
    else:
        optim_img = torch.nn.Parameter(content_img.clone(), requires_grad=True)

    # Initialize optimizer and set image as parameter to be optimized
    optimizer = optim.LBFGS([optim_img])

    # Training Loop
    iter = [0]
    while iter[0] <= num_steps:

        def closure():

            # Set gradients to zero before next optimization step
            optimizer.zero_grad()

            # Clamp image to lie in correct range
            with torch.no_grad():
                optim_img.clamp_(0, 1)

            # Retrieve features of image that is being optimized
            normed_img = normalize(optim_img, vgg_mean, vgg_std)
            input_features = model(normed_img)

            # 2. Calculate the content loss
            if w_content > 0:
                c_loss = w_content * \
                    content_loss(input_features,
                                 content_features, content_layers)
            else:
                c_loss = torch.tensor([0]).to(device)

            # 3a. Calculate the style loss 1
            if w_style_1 > 0:
                s_loss_1 = w_style_1 * \
                    style_loss(input_features, style_features_1, style_layers)
            else:
                s_loss_1 = torch.tensor([0]).to(device)

            # 3b. Calculate the style loss 2
            if w_style_2 > 0:
                s_loss_2 = w_style_2 * \
                    style_loss(input_features, style_features_2, style_layers)
            else:
                s_loss_2 = torch.tensor([0]).to(device)

            # 4. Calculate the total variation loss
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)
            else:
                tv_loss = torch.tensor([0]).to(device)

            # Sum up the losses and do a backward pass
            loss = s_loss_1 + s_loss_2 + c_loss + tv_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % 50 == 0:
                print('iter {}: | Style 1 Loss: {:4f} | Style 2 Loss: {:4f} | Content Loss: {:4f} | TV Loss: {:4f}'.format(
                    iter[0], s_loss_1.item(), s_loss_2.item(), c_loss.item(), tv_loss.item()))

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)

    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    return optim_img
