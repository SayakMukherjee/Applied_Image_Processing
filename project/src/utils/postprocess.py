#----------------------------------------------------------------------------
# Created By  : Sayak Mukherjee
# Created Date: 02-Nov-2022
# version ='1.0'
# ---------------------------------------------------------------------------
# This file contains methods to apply post-processing to the inpainted 
# samples generated by the context-encoder. This is an extension and not
# implemented in the original paper.
# ---------------------------------------------------------------------------

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from utils.config import Config

def poisson_blend(config: Config, images: torch.Tensor, gen_parts: torch.Tensor, topLeft: torch.Tensor, device: str):
    
    images = images.clone().cpu()
    gen_parts = gen_parts.clone().cpu()
    topLeftLoc = topLeft[0].item()

    # Create a mask
    mask = torch.ones_like(gen_parts[0]).cpu()
    mask = transforms.functional.to_pil_image(mask)
    mask = np.array(mask)

    # Find center of in the masked image to place the generated section
    center = (topLeftLoc + config.local_vars['mask_size'] // 2, topLeftLoc + config.local_vars['mask_size'] // 2)

    # Blend masked and generated images
    blended_images = []

    for idx in range(images.shape[0]):

        curr_img = transforms.functional.to_pil_image(images[idx])
        curr_img = np.array(curr_img)

        curr_gen = transforms.functional.to_pil_image(gen_parts[idx])
        curr_gen = np.array(curr_gen)

        out = cv2.seamlessClone(curr_gen, curr_img, mask, center, cv2.MIXED_CLONE)

        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)

        blended_images.append(out)

    blended_images = torch.cat(blended_images, dim=0)

    return blended_images.to(device)
        