import os
from pathlib import Path

import numpy as np

import torch
from torch import Tensor

import vision_transformer as vits
from backbones.BaseBackbone import BaseBackbone


class MedicalViT(BaseBackbone):
    def __init__(self, args):

        if args.backbone.lower() == 'medical_vit_s_8':
            self.patch_size = 8
            if os.name == 'nt':
                self.checkpoint_file = args.backbone_checkpoint_file
            else:
                self.checkpoint_file = args.backbone_checkpoint_file
            channels = [384]
            image_size = 976
            embedding_size = [61]

        elif args.backbone.lower() == 'medical_vit_s_16':
            self.patch_size = 16
            if os.name == 'nt':
                self.checkpoint_file = args.backbone_checkpoint_file
            else:
                self.checkpoint_file = args.backbone_checkpoint_file
            channels = [384]
            image_size = 976
            embedding_size = [61]

        else:
            raise ValueError(f"MedicalViT '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        checkpoint = torch.load(Path(self.checkpoint_file), map_location="cpu")
        teacher_checkpoint = checkpoint['teacher']
        # Discard all weights and parameters belonging to DINOHead() ...
        teacher_dict = {k.replace('backbone.', ''): v for k, v in teacher_checkpoint.items() if k.startswith('backbone.')}

        vit_small = vits.vit_small(patch_size=self.patch_size)
        msg = vit_small.load_state_dict(teacher_dict, strict=True)
        print(f"\nPretrained weights found at '{self.checkpoint_file}'\nand loaded with msg: {msg}")
        
        vit_small.eval() # Activate evaluation mode
        self.backbone = vit_small
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

        return [torch.stack(features)]
