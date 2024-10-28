import os
from pathlib import Path

import math
import numpy as np

import torch
from torch import Tensor

import backbones.vision_transformer as vits
from backbones.BaseBackbone import BaseBackbone


class MedicalViT(BaseBackbone):
    def __init__(self, args):

        if os.name == 'nt':
            self.checkpoint_file = args.backbone_checkpoint_file
        else:
            self.checkpoint_file = args.backbone_checkpoint_file

        self.patch_size = int(args.backbone.lower().split("_")[-1])

        if args.backbone.lower() == 'medical_vit_t_8' or args.backbone.lower() == 'medical_vit_t_16':
            self.model_type = "ViT-T"
            self.image_size = 976 
            self.channels = [192, 192, 192, 192, 192, 192]
            self.embedding_size = [61, 61, 61, 61, 61, 61]
        elif args.backbone.lower() == 'medical_vit_s_8' or args.backbone.lower() == 'medical_vit_s_16':
            self.model_type = "ViT-S"
            self.image_size = 976 
            self.channels = [384, 384, 384, 384, 384, 384]
            self.embedding_size = [61, 61, 61, 61, 61, 61]
        else:
            raise ValueError(f"MedicalViT '{args.backbone}' not supported.")

        super().__init__(num_channels=self.channels, image_size=self.image_size, embedding_size=self.embedding_size, args=args)

    def instantiate(self):

        checkpoint = torch.load(Path(self.checkpoint_file), map_location="cpu", weights_only=False)
        teacher_checkpoint = checkpoint['teacher']
        # Discard all weights and parameters belonging to DINOHead() ...
        teacher_dict = {k.replace('backbone.', ''): v for k, v in teacher_checkpoint.items() if k.startswith('backbone.')}

        if self.model_type.startswith("ViT-T"):
            vit_model = vits.vit_tiny(patch_size=self.patch_size)
        elif self.model_type.startswith("ViT-S"):
            vit_model = vits.vit_small(patch_size=self.patch_size)
        msg = vit_model.load_state_dict(teacher_dict, strict=True)
        print(f"\nPretrained weights found at '{self.checkpoint_file}'\nand loaded with msg: {msg}")
        
        vit_model.eval() # Activate evaluation mode
        self.backbone = vit_model
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

        def reshape_patches(x, patch_size):

            b, _, embed_dim = x.shape                           # [B, num_patches + 1, embed_dim]
            assert (b == B) and (embed_dim == self.embed_dim)

            num_patches = x.shape[1] - 1 # Exclude [CLS] token
            h = w = int(math.sqrt(num_patches))
            assert h * w == num_patches, f"Patches are not forming a square grid! ({h}*{w}!={num_patches})"
            assert h * w == (H / patch_size) * (W / patch_size), "num_patches != (H / patch_size) * (W / patch_size)"

            x = x[:, 1:, :]                                     # Discard [CLS] token -> [B, num_patches, embed_dim]
            x = x.permute(0, 2, 1).contiguous()                 # [B, num_patches, embed_dim] -> [B, embed_dim, num_patches]
            x = x.view(B, embed_dim, h, w)                      # [B, embed_dim, num_patches] -> [B, embed_dim, h, w]

            return x                                            # [B, embed_dim, h, w]

        self.backbone.eval()
        with torch.no_grad():

            intermediate_outputs = self.backbone.get_intermediate_layers(x, n=self.n_layers)
            # intermediate_outputs.shape = n_layers * [B, num_patches + 1, embed_dim]

            extracted_features = [reshape_patches(output, self.patch_size) for output in intermediate_outputs]
            # extracted_features.shape = n_layers * [B, embed_dim, h, w]
            # h, w = (61,61) for ViT-S-16 and image.shape = (976,976)
            
        return extracted_features


    # def forward(self, img_batch: np.ndarray) -> list[Tensor]:

    #     features = []

    #     for img in img_batch:

    #         img = np.expand_dims(img, axis=0)
    #         img = np.repeat(img, 3, axis=0)
    #         img = np.transpose(img, (1, 2, 0))

    #         self.backbone.set_image(img)
    #         features.append(torch.squeeze(self.backbone.get_image_embedding()))

    #     return [torch.stack(features)]
