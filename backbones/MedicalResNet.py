import os
from pathlib import Path

import numpy as np
import torch

from torch import Tensor, nn
from torchvision import models

from backbones.BaseBackbone import BaseBackbone


class MedicalResNet(BaseBackbone):
    def __init__(self, args):

        if args.backbone.lower() == 'medical_resnet50':
            if os.name == 'nt':
                self.checkpoint_file = args.backbone_checkpoint_file
            else:
                self.checkpoint_file = args.backbone_checkpoint_file
            image_size = 976 
            channels = [256, 512, 1024, 2048]
            embedding_size = [244, 124, 62, 31]

        else:
            raise ValueError(f"MedicalResNet '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        checkpoint = torch.load(Path(self.checkpoint_file), map_location="cpu")
        teacher_checkpoint = checkpoint['teacher']
        # Discard all weights and parameters belonging to DINOHead() ...
        teacher_dict = {k.replace('module.backbone.', ''): v for k, v in teacher_checkpoint.items() if k.startswith('module.backbone.')}

        resnet50 = models.resnet50(weights=None)
        resnet50.fc = nn.Identity() # Needed to successfully load checkpoint
        msg = resnet50.load_state_dict(teacher_dict, strict=True)
        print(f"\nPretrained weights found at '{self.checkpoint_file}'\nand loaded with msg: {msg}")
        resnet50 = nn.Sequential(*list(resnet50.children())[:-1]) # Discard the last FC layer (only needed for class predictions)

        resnet50.eval() # Activate evaluation mode
        self.backbone = resnet50
        self.backbone = self.backbone.cuda()

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
