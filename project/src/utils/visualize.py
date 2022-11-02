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

from torchvision.utils import save_image
from torch.autograd import Variable
from networks import ContextEncoder
from utils.config import Config

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
    batch_size =12
    # Get data loaders
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

    generated_images = masked_images.clone()

    generated_images[:, :, 
                     topLeftLoc : topLeftLoc + config.local_vars['mask_size'], 
                     topLeftLoc : topLeftLoc + config.local_vars['mask_size']] = gen_parts

    # Save results
    sample = torch.cat((masked_images.data, generated_images.data, images.data), -1)
    save_image(sample, "../outputs/%s.png" % name, nrow=3, normalize=True)    

    logger.info('Completed generating visuals...')