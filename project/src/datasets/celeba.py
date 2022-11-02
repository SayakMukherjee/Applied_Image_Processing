#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains code for configuring the CelebA dataset. Prior to 
# execution make sure to download the celebA align dataset from
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and extract the contents
# into the ../data/ directory. Randomly choosen 14900 images are being
# used to train context encoder owing to resource constraints.
# ---------------------------------------------------------------------------

import os
import json
import numpy as np
import torchvision.transforms as transforms
import logging
import random

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


class CelebADataset():
    """Base class for CelebA Dataset
    """

    def __init__(self, root:str, image_size:int, mask_size:int):

        self.logger = logging.getLogger()

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = CelebA(root, self.transform, mode='train', mask_size = mask_size, image_size = image_size)

        self.val_set = CelebA(root, self.transform, mode='val', mask_size = mask_size, image_size = image_size)

        self.test_set = CelebA(root, self.transform, mode='test', mask_size = mask_size, image_size = image_size)

        self.logger.info('Train: %d samples, Val: %d samples, Test: %d samples'
                        % (len(self.train_set), len(self.val_set), len(self.test_set)))

    def loaders(self, batch_size: int, shuffle: bool =True, num_workers: int = 0):
        """Method to create train, test and validation dataloaders

        Args:
            batch_size (int): batch size
            shuffle (bool, optional): shuffle the order of samples. Defaults to True.
            num_workers (int, optional): number of concurrent workers. Defaults to 0.

        Returns:
            train_dataLoader, val_dataLoader, test_dataLoader (DataLoader): dataloaders
        """

        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)

        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)

        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)

        return train_dataLoader, val_dataLoader, test_dataLoader


class CelebA(Dataset):
    """Implementation of CelebA Dataset
    """

    def __init__(self, root: str, transform: transforms, mask_size:int, image_size:int, mode: str):
        super().__init__()

        self.root = root
        self.mode = mode
        self.transform = transform
        self.image_size = image_size
        self.mask_size = mask_size
        self.imagePath = os.path.join(self.root, 'img_align_celeba')

        # Train-test split
        if not os.path.exists(os.path.join(self.root, 'train-test-split.json')):
            self.__configure_dataset__()

        with open(os.path.join(self.root, 'train-test-split.json'), 'r') as file:
            self.train_test_split = json.load(file)

        if self.mode in ['train', 'val']:

            self.images = self.train_test_split["train"]

            train_size = (int) (len(self.images) * 0.9)

            if self.mode == 'train':
                self.images = self.images[:train_size]
            else:
                self.images = self.images[train_size:]

        else:

            self.images = self.train_test_split["test"]

    def __configure_dataset__(self):
        """Initial configuration for dividing samples
        """

        all_images = os.listdir(self.imagePath)

        train_split = random.sample(all_images, 14900)

        for elems in train_split:
            all_images.remove(elems)

        test_split = random.sample(all_images, 100)

        with open(os.path.join(self.root, 'train-test-split.json'), 'w') as fp:
            json.dump({"train": train_split, "test": test_split}, fp)


    def random_mask(self, image):
        """Generate a mask at a random location

        Args:
            image (Tensor): image to apply mask on

        Returns:
            masked_image (Tensor): image with mask
            masked_section (Tensor): section of the image that is masked
        """
        x0, y0 = np.random.randint(0, self.image_size - self.mask_size, 2)
        x1, y1 = x0 + self.mask_size, y0 + self.mask_size

        masked_section = image[:, y0:y1, x0:x1]
        masked_image = image.clone()
        masked_image[:, y0:y1, x0:x1] = 1

        return masked_image, masked_section

    def centre_mask(self, image):
        """Generate a mask at the centre of the image

        Args:
            image (Tensor): image to apply mask on

        Returns:
            masked_image (Tensor): image with mask
            masked_section (Tensor): section of the image that is masked
        """
        x0 = (self.image_size - self.mask_size) // 2
        x1 = x0 + self.mask_size

        masked_section = image[:, x0:x1, x0:x1]
        masked_image = image.clone()
        masked_image[:, x0:x1, x0:x1] = 1

        return masked_image, masked_section, x0


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        image = Image.open(os.path.join(self.imagePath, self.images[index]))

        image = self.transform(image)

        if self.mode == 'train':
            masked_image, masked_section = self.random_mask(image)
            return image, masked_image, masked_section

        else:
            masked_image, masked_section, topLeft = self.centre_mask(image)
            return image, masked_image, topLeft