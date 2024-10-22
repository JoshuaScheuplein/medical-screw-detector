import os
from pathlib import Path

import numpy as np

import torch
import torchvision.transforms as T
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
                
            self.image_size = 976 
            self.channels = [256, 512, 1024, 2048]
            self.embedding_size = [244, 122, 61, 31]

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

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone = self.backbone.cuda()

    '''
    Input image batch of shape (B, C, H, W) 
    Return the embeddings of the image as list of tensors of shape (B, num_channel, embedding_size, embedding_size)
    '''
    def forward(self, img_batch: np.ndarray) -> list[Tensor]:

        transformed_images = []

        for img in img_batch:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            transform = T.Compose([

                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255]
                # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                # if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
                # or if the numpy.ndarray has dtype = np.uint8
                T.ToTensor(),

                # Note: DINO ResNet was trained on np.float32 images in range [0.0,1.0]
                # lambda x: 255.0 * x,  # scale by 255

            ])

            transformed_images.append(transform(img))

        img_batch = torch.stack(transformed_images)
        img_batch = img_batch.to(self.args.device)

        self.backbone.eval()
        extracted_features = []
        with torch.no_grad():
            x = img_batch
            for i, layer in enumerate(self.backbone):
                x = layer(x)
                if 4 <= i and i <= 7:
                    assert x.shape[1:] == (self.channels[i-4], self.embedding_size[i-4], self.embedding_size[i-4])
                    extracted_features.append(x)

        return extracted_features


    # '''
    #     Input image batch of shape (B, C, H, W) 
    #     Return the embeddings of the image as list of tensors of shape (B, num_channel, embedding_size, embedding_size)
    # '''
    # def forward(self, img_batch: np.ndarray) -> list[Tensor]:

    #     features = []

    #     for img in img_batch:

    #         img = np.expand_dims(img, axis=0)
    #         img = np.repeat(img, 3, axis=0)
    #         img = np.transpose(img, (1, 2, 0))

    #         self.backbone.set_image(img)
    #         features.append(torch.squeeze(self.backbone.get_image_embedding()))

    #         # from head_tip_segmentation.scripts.check_sam import plot_pca_preview
    #         # plot_pca_preview(img, features[0], 256, 64, show=True)

    #     return [torch.stack(features)]
