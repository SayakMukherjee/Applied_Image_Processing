import logging
import torch
import os

from torchvision.utils import save_image
from torch.autograd import Variable
from networks import ContextEncoder
from utils.config import Config

from torch.utils.data import Dataset

def visualize_samples(config: Config, dataset: Dataset, generator: ContextEncoder, device: str, name: str):

    logger = logging.getLogger()

    logger.info('Start generating visuals...')

    # Create log directory
    if not os.isdir(config.local_vars['save_path']):
        os.mkdir(config.local_vars['save_path'])

    # Get data loaders
    _, _, test_dataLoader = dataset.loaders(batch_size = 8)

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
    sample = torch.cat((masked_images.data, generated_images.data, images.data), -2)
    save_image(sample, "../images/%s.png" % name, nrow=6, normalize=True)

    logger.info('Completed generating visuals...')