import os

import numpy as np
import torch

from segment_anything import SamPredictor, sam_model_registry
from torch import Tensor

from backbones.BaseBackbone import BaseBackbone


class SAMBackbone(BaseBackbone):
    def __init__(self, args):

        if args.backbone.lower() == 'sam_vit_b':
            if os.name == 'nt':
                self.src = rf"C:\Users\wagne\Downloads\sam_vit_b_01ec64.pth"
            else:
                self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/sam_vit_b_01ec64.pth"
            channels = [256]
            image_size = 976
            embedding_size = [64]

        elif args.backbone.lower() == 'sam_vit_l':
            if os.name == 'nt':
                self.src = rf"C:\Users\wagne\Downloads\sam_vit_l_0b3195.pth"
            else:
                self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/sam_vit_l_0b3195.pth"
            channels = [256]
            image_size = 976
            embedding_size = [64]

        elif args.backbone.lower() == 'sam_vit_h':
            if os.name == 'nt':
                self.src = rf"C:\Users\wagne\Downloads\sam_vit_h_4b8939.pth"
            else:
                self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/sam_vit_h_4b8939.pth"
            channels = [256]
            image_size = 976
            embedding_size = [64]

        else:
            raise ValueError(f"SAM Backbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        sam = sam_model_registry[self.args.backbone.lower()[4:]](checkpoint=self.src)
        sam.to(self.args.device)
        self.backbone = SamPredictor(sam)

    '''
        Input image batch of shape (B, C, H, W) 
        Return the embeddings of the image as list of tensors of shape (B, num_channel, embedding_size, embedding_size)
    '''
    def forward(self, img_batch: np.ndarray) -> list[Tensor]:

        features = []

        for img in img_batch:

            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            self.backbone.set_image(img)
            features.append(torch.squeeze(self.backbone.get_image_embedding()))

        return [torch.stack(features)]
