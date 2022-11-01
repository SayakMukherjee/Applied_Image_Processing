from asyncio.log import logger
import os
import numpy as np
import torchvision.transforms as transforms
import logging

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split


class ParisStreetViewDataset():

    def __init__(self, root:str, image_size:int, mask_size:int):

        self.logger = logging.getLogger()

        self.transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_set = ParisStreetView(root, self.transform, mode='train', mask_size = mask_size, image_size = image_size)

        self.val_set = ParisStreetView(root, self.transform, mode='val', mask_size = mask_size, image_size = image_size)

        self.test_set = ParisStreetView(root, self.transform, mode='test', mask_size = mask_size, image_size = image_size)

        self.logger.info('Train: %d samples, Val: %d samples, Test: %d samples' 
                        % (len(self.train_set), len(self.val_set), len(self.test_set)))

    def loaders(self, batch_size: int, shuffle: bool =True, num_workers: int = 0):

        train_dataLoader = DataLoader(self.train_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)
        
        val_dataLoader = DataLoader(self.val_set, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers)

        test_dataLoader = DataLoader(self.test_set, batch_size=batch_size,
                         shuffle=shuffle, num_workers=num_workers)

        return train_dataLoader, val_dataLoader, test_dataLoader


class ParisStreetView(Dataset):

    def __init__(self, root: str, transform: transforms, mask_size:int, image_size:int, mode: str):
        super().__init__()

        self.root = root
        self.mode = mode
        self.transform = transform
        self.image_size = image_size
        self.mask_size = mask_size

        if self.mode in ['train', 'val']:
            self.imagePath = os.path.join(self.root, 'paris_train_original')

            self.images = os.listdir(self.imagePath)

            train_size = (int) (len(self.images) * 0.9)

            if self.mode == 'train':
                self.images = self.images[:train_size]
            else:
                self.images = self.images[train_size:]

        else:
            self.imagePath = os.path.join(self.root, 'paris_eval_gt')

            self.images = os.listdir(self.imagePath)

    def random_mask(self, image):
        x0, y0 = np.random.randint(0, self.image_size - self.mask_size, 2)
        x1, y1 = x0 + self.mask_size, y0 + self.mask_size

        masked_section = image[:, y0:y1, x0:x1]
        masked_image = image.clone()
        masked_image[:, y0:y1, x0:x1] = 1

        return masked_image, masked_section

    def centre_mask(self, image):
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