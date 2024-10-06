import os

import numpy as np
import torch

from torch import Tensor
from backbones.BaseBackbone import BaseBackbone


class SAM2Backbone(BaseBackbone):
    def __init__(self, args):

        if args.backbone.lower() == 'sam2_large':
            if os.name == 'nt':
                self.src = rf"C:\Users\wagne\Downloads\sam2_hiera_large.pt"
            else:
                self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/sam2_hiera_large.pth"
            self.cfg = "sam2_hiera_l.yaml"
            channels = [256]
            image_size = 976
            embedding_size = [64]

        elif args.backbone.lower() == 'sam2_base':
            if os.name == 'nt':
                self.src = rf"C:\Users\wagne\Downloads\sam2_hiera_base_plus.pt"
            else:
                self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/sam2_hiera_base_plus.pth"
            self.cfg = "sam2_hiera_b+.yaml"
            channels = [256]
            image_size = 976
            embedding_size = [64]

        else:
            raise ValueError(f"SAM Backbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.backbone = SAM2ImagePredictor(build_sam2(self.cfg, self.src))
        pass

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

            # from head_tip_segmentation.scripts.check_sam import plot_pca_preview
            # plot_pca_preview(img, features[0], 256, 64, show=True)

        return [torch.stack(features)]
