#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains code for training and testing the context encoder.
# ---------------------------------------------------------------------------

import logging
import random
import numpy as np
import torch
import os

from torch.autograd import Variable
from datasets import ParisStreetViewDataset, CelebADataset
from networks import ContextEncoder, Discriminator
from utils.config import Config
from utils.visualize import visualize_samples
from ignite.metrics import PSNR, SSIM
from ignite.engine import Engine
from torch.utils.data import Dataset

################################
# Utility functions for training
################################

def save_model(config: Config, generator: ContextEncoder, discriminator: Discriminator, name: str):
    """Method to save the trained models

    Args:
        config (Config): config object
        generator (ContextEncoder): generator model
        discriminator (Discriminator): discriminator model
        name (str): name of the saved model file
    """

    if not os.path.isdir(config.local_vars['model_path']):
        os.mkdir(config.local_vars['model_path'])

    generator_dict = generator.state_dict()
    discriminator_dict = discriminator.state_dict()

    export_path = os.path.join(config.local_vars['model_path'], name + '.tar')

    torch.save({'generator_dict': generator_dict,
                'discriminator_dict': discriminator_dict}, export_path)

def load_model(config: Config, generator: ContextEncoder, discriminator: Discriminator, name: str):
    """Method to load saved model

    Args:
        config (Config): config object
        generator (ContextEncoder): generator model
        discriminator (Discriminator): discriminator model
        name (str): name of the saved model file

    Returns:
        generator (ContextEncoder): loaded generator model
        discriminator (Discriminator): loaded discriminator model
    """

    import_path = os.path.join(config.local_vars['model_path'], name + '.tar')

    model_dict = torch.load(import_path)

    generator.load_state_dict(model_dict['generator_dict'])
    discriminator.load_state_dict(model_dict['discriminator_dict'])

    return generator, discriminator

def init_weights(model):
    """Method to initialise the model weights

    Args:
        model (ContextEncoder or Discriminator): model whose weights to be initialised
    """

    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        torch.nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(model.bias.data, 0.0)

###########################
# End of Utility functions 
###########################

def train(config: Config, dataset: Dataset, generator: ContextEncoder, discriminator: Discriminator, device: str):
    """Method to train the generator and discriminator

    Args:
        config (Config): config object
        dataset (Dataset): dataset object
        generator (ContextEncoder): generator model
        discriminator (Discriminator): discriminator model
        device (str): 'cuda' or 'cpu'

    """

    logger = logging.getLogger()

    # Disable info logs for ignite engine
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    # Initialising model weights
    generator.apply(init_weights)
    discriminator.apply(init_weights)

    logger.info('Models initialised')

    # Loss functions
    adv_loss = torch.nn.BCELoss()
    pixel_loss = torch.nn.MSELoss()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(),
                                   lr = config.local_vars['learning_rate'] * 10,
                                   betas = (config.local_vars['b1'], config.local_vars['b2']))

    optimizer_D = torch.optim.Adam(discriminator.parameters(),
                                   lr = config.local_vars['learning_rate'],
                                   betas = (config.local_vars['b1'], config.local_vars['b2']))

    # Get data loaders
    train_loader, val_dataLoader, _ = dataset.loaders(batch_size = config.local_vars['batch_size'])

    # Size of the output of image discriminator
    out_height, out_width = int(config.local_vars['mask_size'] / 2 ** 3), int(config.local_vars['mask_size'] / 2 ** 3)
    out_size = (1, out_height, out_width)

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # Additional metrics
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')

    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, 'ssim')

    logger.info('Start training...')
    for epoch in range(config.local_vars['epochs']):
        for i, (images, masked_images, masked_sections) in enumerate(train_loader):

            # True and fake labels
            valid_labels = Variable(Tensor(images.shape[0], *out_size).fill_(1.0), requires_grad=False)
            fake_labels = Variable(Tensor(images.shape[0], *out_size).fill_(0.0), requires_grad=False)

            # Configure input
            images = Variable(images.type(Tensor))
            masked_images = Variable(masked_images.type(Tensor))
            masked_sections = Variable(masked_sections.type(Tensor))

            # Train generator

            optimizer_G.zero_grad()

            # Generate patches using the generator
            gen_parts = generator(masked_images)

            # Calculate adversarial and pixelwise loss
            gen_adv = adv_loss(discriminator(gen_parts), valid_labels)
            gen_pixel = pixel_loss(gen_parts, masked_sections)

            # Total loss of generator
            gen_loss = 0.001 * gen_adv + 0.999 * gen_pixel

            gen_loss.backward()
            optimizer_G.step()

            # Train discriminator

            optimizer_D.zero_grad()

            # Calculate adversarial loss for discriminator
            real_loss = adv_loss(discriminator(masked_sections), valid_labels)
            fake_loss = adv_loss(discriminator(gen_parts.detach()), fake_labels)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            logger.info(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                % (epoch, config.local_vars['epochs'], i, len(train_loader), d_loss.item(), gen_adv.item(), gen_pixel.item())
            )

        # Validate training
        if epoch % config.local_vars['val_interval'] == 0:

            psnr_loss = 0
            ssim_loss = 0
            total = 0

            for i, (images, masked_images, topLeft) in enumerate(val_dataLoader):

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


                # Calculate metrics
                state = default_evaluator.run([[generated_images, images]])

                psnr_loss += state.metrics['psnr']
                ssim_loss += state.metrics['ssim']
                total += images.shape[0]

            logger.info(
                "[Validation] [PSNR loss: %f] [SSIM loss: %f]"
                % (psnr_loss/total, ssim_loss/total )
            )

        # Save model checkpoints
        if epoch % config.local_vars['chkpoint_interval'] == 0 and epoch != config.local_vars['epochs'] - 1:
            save_model(config, generator, discriminator, 'model_' + config.local_vars['dataset'] + '_' +str(epoch))

    logger.info('Completed training...')

def test(config: Config, dataset: Dataset, generator: ContextEncoder, device: str):
    """Method to test the trained generator and discriminator

    Args:
        config (Config): config object
        dataset (Dataset): dataset object
        generator (ContextEncoder): generator model
        discriminator (Discriminator): discriminator model
        device (str): 'cuda' or 'cpu'

    """
    logger = logging.getLogger()

    # Disable info logs for ignite engine
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    logger.info('Start testing...')

    # Get data loaders
    _, _, test_dataLoader = dataset.loaders(batch_size = config.local_vars['batch_size'])

    # Initialise loss
    psnr_loss = 0
    ssim_loss = 0
    total = 0

    Tensor = torch.cuda.FloatTensor if device == 'cuda' else torch.FloatTensor

    # Additional metrics
    def eval_step(engine, batch):
        return batch

    default_evaluator = Engine(eval_step)

    psnr = PSNR(data_range=1.0)
    psnr.attach(default_evaluator, 'psnr')

    ssim = SSIM(data_range=1.0)
    ssim.attach(default_evaluator, 'ssim')

    for i, (images, masked_images, topLeft) in enumerate(test_dataLoader):

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

        # Calculate metrics
        state = default_evaluator.run([[generated_images, images]])

        psnr_loss += state.metrics['psnr']
        ssim_loss += state.metrics['ssim']
        total += images.shape[0]

    logger.info(
        "[Test] [PSNR loss: %f] [SSIM loss: %f]"
        % (psnr_loss/total, ssim_loss/total )
    )

    logger.info('Completed testing...')

def main():

    # Load configuration
    config = Config(locals().copy())
    config.load_config(import_path='config.json')

    # Create log directory
    if not os.path.isdir(config.local_vars['log_path']):
        os.mkdir(config.local_vars['log_path'])

    # Setup logger
    logging.basicConfig(level=logging.INFO,
                        filename=config.local_vars['log_path'] + '/log.txt',
                        filemode='w',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger()

    logger.info('Log file is %s.' % (config.local_vars['log_path'] + '/log.txt'))
    logger.info('Data path is %s.' % config.local_vars['data_path'])

    # Set seed
    if config.local_vars['seed'] != -1:

        # if -1 then keep randomised
        random.seed(config.local_vars['seed'])
        np.random.seed(config.local_vars['seed'])
        torch.manual_seed(config.local_vars['seed'])
        logger.info('Set seed to %d.' % config.local_vars['seed'])

    # Check device type
    if torch.cuda.is_available() and config.local_vars['device'] == 'cuda':
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info('Computation device: %s' % device)

    # Configure dataset
    if config.local_vars['dataset'] == "parisStreetView":
        dataset = ParisStreetViewDataset(root = config.local_vars['data_path'],
                                        image_size = config.local_vars['image_size'],
                                        mask_size = 64)

    elif config.local_vars['dataset'] == "celeba":
        dataset = CelebADataset(root = config.local_vars['data_path'],
                                image_size = config.local_vars['image_size'],
                                mask_size = 64)
    else:
        logger.info('Unknown dataset')
        return

    logger.info('Configured dataset: %s.' %config.local_vars['dataset'])

    # Generator and discriminator models
    generator = ContextEncoder().to(device)
    discriminator = Discriminator().to(device)

    # Train, test and visualise samples
    train(config, dataset, generator, discriminator, device)

    test(config, dataset, generator, device)

    visualize_samples(config, dataset, generator, device, 'Results_' + config.local_vars['dataset'])

    # Save trained models
    save_model(config, generator, discriminator, 'model_' + config.local_vars['dataset'])

if __name__ == '__main__':
    main()