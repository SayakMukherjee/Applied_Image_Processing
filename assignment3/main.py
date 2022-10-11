import os
import torch
import torch.optim as optim

from helper_functions import *
from your_code_here import *

os.environ['TORCH_HOME'] = 'models'

torch.manual_seed(2022)  # Set random seed for better reproducibility
device = 'cpu'  # Make sure that if you use cuda that it also runs on CPU

# Hyperparameters
img_size = 128
# Sets of hyperparameters that worked well for us
if img_size == 128:
    num_steps = 1000
    w_style_1 = 1e5
    w_style_2 = 1e5
    w_content = 1
    w_tv = 5e-4
else:
    num_steps = 1000
    w_style_1 = 1e6
    w_style_2 = 1e6
    w_content = 1
    w_tv = 5e-6

# Choose what feature maps to extract for the content and style loss
# We use the ones as mentioned in Gatys et al. 2016
content_layers = ['conv4_2']
style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

# Paths
out_folder = 'outputs'
style_img_path_1 = os.path.join('data', 'gogh.jpg')
style_img_path_2 = os.path.join('data', 'munch.jpg')
content_img_path = os.path.join('data', 'duck.jpg')

# Load style and content images as resized (squared) tensors
style_img_1 = image_loader(style_img_path_1, device=device, img_size=img_size)
style_img_2 = image_loader(style_img_path_2, device=device, img_size=img_size)
content_img = image_loader(content_img_path, device=device, img_size=img_size)

# Define the channel-wise mean and standard deviation used for VGG training
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def run_single_image(vgg_mean, vgg_std, content_img, style_img, num_steps=num_steps,
                     random_init=True, w_style=w_style_1, w_content=w_content, w_tv=w_tv):
    """ Neural Style Transfer optmization procedure for a single style image.

    # Parameters:
        @vgg_mean, VGG channel-wise mean, torch.tensor of size (c)
        @vgg_std, VGG channel-wise standard deviation, detorch.tensor of size (c)
        @content_img, torch.tensor of size (1, c, h, w)
        @style_img, torch.tensor of size (1, c, h, w)
        @num_steps, int, iteration steps
        @random_init, bool, whether to start optimizing with based on a random image. If false,
            the content image is as initialization.
        @w_style, float, weight for style loss
        @w_content, float, weight for content loss 
        @w_tv, float, weight for total variation loss

    # Returns the style-transferred image
    """

    # Initialize Model
    model = Vgg19(content_layers, style_layers, device)

    # TODO: 1. Normalize Input images
    normed_style_img = normalize(style_img, vgg_mean, vgg_std)
    normed_content_img = normalize(content_img, vgg_mean, vgg_std)

    # Retrieve feature maps for content and style image
    style_features = model(normed_style_img)
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

            # TODO: 2. Calculate the content loss
            if w_content > 0:
                c_loss = w_content * \
                    content_loss(input_features,
                                 content_features, content_layers)
            else:
                c_loss = torch.tensor([0]).to(device)

            # TODO: 3. Calculate the style loss
            if w_style > 0:
                s_loss = w_style * \
                    style_loss(input_features, style_features, style_layers)
            else:
                s_loss = torch.tensor([0]).to(device)

            # TODO: 4. Calculate the total variation loss
            if w_tv > 0:
                tv_loss = w_tv * total_variation_loss(normed_img)
            else:
                tv_loss = torch.tensor([0]).to(device)

            # Sum up the losses and do a backward pass
            loss = s_loss + c_loss + tv_loss
            loss.backward()

            # Print losses every 50 iterations
            iter[0] += 1
            if iter[0] % 50 == 0:
                print('iter {}: | Style Loss: {:4f} | Content Loss: {:4f} | TV Loss: {:4f}'.format(
                    iter[0], s_loss.item(), c_loss.item(), tv_loss.item()))

            return loss

        # Do an optimization step as defined in our closure() function
        optimizer.step(closure)

    # Final clamping
    with torch.no_grad():
        optim_img.clamp_(0, 1)

    return optim_img


# Single image optimization
print('Start single style image optimization.')
output1 = run_single_image(
    vgg_mean, vgg_std, content_img, style_img_1, num_steps=num_steps,
    random_init=True, w_style=w_style_1, w_content=w_content, w_tv=w_tv)
output_name1 = f'single img_size-{img_size} num_steps-{num_steps} w_style-{w_style_1} w_content-{w_content} w_tv-{w_tv}'
save_image(output1, title=output_name1, out_folder=out_folder)

# Mixing of multiple style images
# TODO: 5. Implement style transfer for two given images
# print('\nStart double style image optimization.')
# output2 = run_double_image(
#     vgg_mean, vgg_std, content_img, style_img_1, style_img_2, num_steps=num_steps,
#     random_init=True, w_style_1=w_style_1, w_style_2=w_style_2, w_content=w_content, w_tv=w_tv,
#     content_layers=content_layers, style_layers=style_layers, device=device)
# output_name2 = f'double img_size-{img_size} num_steps-{num_steps} w_style_1-{w_style_1} w_style_2-{w_style_2} w_content-{w_content} w_tv-{w_tv}'
# save_image(output2, title=output_name2, out_folder=out_folder)
