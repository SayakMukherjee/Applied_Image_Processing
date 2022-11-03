#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains the methods to visualise the results
# ---------------------------------------------------------------------------

import logging
import torch
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torch.autograd import Variable
from networks import ContextEncoder
from utils.config import Config
from utils.postprocess import poisson_blend
from torch.utils.data import Dataset

def visualize_samples(config: Config, dataset: Dataset, generator: ContextEncoder, device: str, name: str):
    """Visualise results by plotting ground truth, masked image and inpainted image

    Args:
        config (Config): config object
        dataset (Dataset): dataset object
        generator (ContextEncoder): generator model
        device (str): 'cuda' or 'cpu'
        name (str): name for saving the visualization
    """

    logger = logging.getLogger()

    logger.info('Start generating visuals...')

    # Create log directory
    if not os.path.isdir(config.local_vars['save_path']):
        os.mkdir(config.local_vars['save_path'])

    # Get data loaders
    batch_size = 12
    _, _, test_dataLoader = dataset.loaders(batch_size = batch_size)

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    images, masked_images, topLeft = next(iter(test_dataLoader))

    # Configure input
    images = Variable(images.type(Tensor))
    masked_images = Variable(masked_images.type(Tensor))

    topLeftLoc = topLeft[0].item()

    # Inpaint samples
    with torch.no_grad():
        gen_parts = generator(masked_images)

    if config.local_vars['postprocess']:
        generated_images_blended = poisson_blend(config, masked_images, gen_parts, topLeft, device)


    generated_images = masked_images.clone()

    generated_images[:, :, 
                    topLeftLoc : topLeftLoc + config.local_vars['mask_size'], 
                    topLeftLoc : topLeftLoc + config.local_vars['mask_size']] = gen_parts

    # Save results
    if config.local_vars['postprocess']:
        sample = torch.cat((masked_images.data, generated_images.data, generated_images_blended.data, images.data), -1)
        save_image(sample, "../outputs/%s.png" % name, nrow=3, normalize=True)
    else:  

        fig, ax = plt.subplots(4, 9, figsize=(30,15))

        nrow = 0

        for idx in range(0, batch_size, 3):
            # 1st image
            img = masked_images[idx].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][0].imshow(img)
            ax[nrow][0].axis('off')
            ax[nrow][0].set_title('Masked Image')

            img = generated_images[idx].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][1].imshow(img)
            ax[nrow][1].axis('off')
            ax[nrow][1].set_title('Inpainted Image')

            img = images[idx].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][2].imshow(img)
            ax[nrow][2].axis('off')
            ax[nrow][2].set_title('Original Image')

            # 2nd image
            img = masked_images[idx + 1].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][3].imshow(img)
            ax[nrow][3].axis('off')
            ax[nrow][3].set_title('Masked Image')

            img = generated_images[idx + 1].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][4].imshow(img)
            ax[nrow][4].axis('off')
            ax[nrow][4].set_title('Inpainted Image')

            img = images[idx + 1].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][5].imshow(img)
            ax[nrow][5].axis('off')
            ax[nrow][5].set_title('Original Image')

            # 3rd image
            img = masked_images[idx + 2].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][6].imshow(img)
            ax[nrow][6].axis('off')
            ax[nrow][6].set_title('Masked Image')

            img = generated_images[idx + 2].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][7].imshow(img)
            ax[nrow][7].axis('off')
            ax[nrow][7].set_title('Inpainted Image')

            img = images[idx + 2].cpu().clone() * torch.Tensor([0.5]) + torch.Tensor([0.5])
            img = transforms.ToPILImage()(img)
            ax[nrow][8].imshow(img)
            ax[nrow][8].axis('off')
            ax[nrow][8].set_title('Original Image')

            nrow += 1

        fig.tight_layout()

        plt.xlabel('')
        plt.ylabel('')

        fig.savefig("../outputs/%s.png" % name)

    logger.info('Completed generating visuals...')