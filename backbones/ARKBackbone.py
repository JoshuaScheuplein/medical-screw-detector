import os

import numpy as np
import timm

import torch
import torchvision.transforms as T
from torch import Tensor

from backbones.BaseBackbone import BaseBackbone


class ARKBackbone(BaseBackbone):
    def __init__(self, args):
        image_size = 1024

        if args.backbone == 'ark5_low_res':
            channels = [256, 512, 1024]
            embedding_size = [112, 56, 28]
            self.ark_5_src()
        elif args.backbone == 'ark5':
            channels = [128, 256, 512, 1024]
            embedding_size = [224, 112, 56, 28]
            self.ark_5_src()
        elif args.backbone == 'ark5_single':
            channels = [256]
            embedding_size = [112]
            self.ark_5_src()
        elif args.backbone == 'ark6_low_res':
            channels = [256, 512, 1024]
            embedding_size = [112, 56, 28]
            self.ark_6_src()
        elif args.backbone == 'ark6_very_low_res':
            channels = [512, 1024]
            embedding_size = [56, 28]
            self.ark_6_src()
        elif args.backbone == 'ark6':
            channels = [128, 256, 512, 1024]
            embedding_size = [224, 112, 56, 28]
            self.ark_6_src()
        elif args.backbone == 'ark6_single':
            channels = [1024]
            embedding_size = [28]
            self.ark_6_src()
        else:
            raise ValueError(f"ArkBackbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        timm_model = timm.create_model('swin_base_patch4_window7_224', num_classes=1376, img_size=1024, pretrained=False, features_only=True)

        state_dict = torch.load(self.src, map_location="cpu")
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in state_dict:
                del state_dict[k]

        converted_dict = {}
        for k, v in state_dict.items():
            import re
            k = re.sub(r'layers.(\d+).downsample', lambda x: f'layers.{int(x.group(1)) + 1}.downsample', k)
            k = re.sub(r'layers\.', 'layers_', k)

            converted_dict[k] = v

        intersection = list(set(converted_dict.keys()) & set(timm_model.state_dict().keys()))
        print(intersection)

        for k in timm_model.state_dict().keys():
            if k not in converted_dict.keys():
                print(k)

        timm_model.load_state_dict(converted_dict, strict=False)

        self.backbone = timm_model.cuda()

    def forward(self, img_batch: np.ndarray) -> list[Tensor]:
        b, _W, _H = img_batch.shape

        transformed_patches = []

        for img in img_batch:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            transform = T.Compose([
                T.ToTensor(),
                lambda x: 255.0 * x,  # scale by 255
            ])

            # # Split the image into 4 chunks along the 2nd dimension (height)
            # patches_rows = torch.chunk(transform(img), 4, dim=2)
            # # Split each chunk into 4 chunks along the 3rd dimension (width)
            # patches = [torch.chunk(chunk, 4, dim=1) for chunk in patches_rows]
            # # Flatten the list of lists into a single list of 16 tensors
            # patches = [item for sublist in patches for item in sublist]

            transformed_patches.append(transform(img))

        #     patches = list(transform(img).permute(1, 2, 0).unfold(0, 224, 224).unfold(1, 224, 224).reshape(16, 3, 224, 224))
        #     transformed_patches += patches
        #
        img_batch = torch.stack(transformed_patches)
        img_batch = img_batch.to(self.args.device)

        layers = self.backbone(img_batch)

        # output = []
        #
        # for i, layer in enumerate(layers):
        #
        #     patch_size = self.embedding_size[i] // 4
        #     embedding_size = self.embedding_size[i]
        #     channels = self.num_channels[i]
        #
        #     # tmp = []
        #     # for embeddings in layer.view(b, 16, patch_size, patch_size, channels):
        #     #     embeddings = torch.unsqueeze(embeddings, 1)
        #     #     embeddings = [embedding.permute(0, 3, 1, 2) for embedding in embeddings]
        #     #
        #     #     # Reshape the list into a 4x4 grid
        #     #     grid = [embeddings[i:i + 4] for i in range(0, 16, 4)]
        #     #
        #     #     # Concatenate the tensors in each row
        #     #     rows = [torch.cat(row_tensors, dim=2) for row_tensors in grid]
        #     #
        #     #     # Concatenate the rows vertically
        #     #     result = torch.cat(rows, dim=3)
        #     #
        #     #     tmp.append(result.squeeze())
        #     #
        #     # output.append(torch.stack(tmp))
        #
        #     # Reshape the layer into a 4x4 grid of patches
        #     layer = layer.reshape(b, 4, 4, patch_size, patch_size, channels)
        #     # Transpose the patches to the correct order
        #     layer = layer.permute(0, 5, 3, 4, 1, 2)
        #     # Reshape the patches into a 2D tensor
        #     layer = layer.reshape(b, patch_size * patch_size * channels, 16)
        #
        #     # Fold the patches back into a single tensor
        #     layer = torch.nn.functional.fold(layer,
        #                                      output_size=(embedding_size, embedding_size),
        #                                      kernel_size=(patch_size, patch_size),
        #                                      stride=(patch_size, patch_size))
        #
        #     output.append(layer)

        return [layer.permute(0, 3, 1, 2) for layer in layers[4 - len(self.embedding_size):]]

    def ark_6_src(self):
        if os.name == 'nt':
            self.src = rf"C:\Users\wagne\Downloads\ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar"
        else:
            self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/ark6_teacher_ep200_swinb_projector1376_mlp.pth.tar"

    def ark_5_src(self):
        if os.name == 'nt':
            self.src = rf"C:\Users\wagne\Downloads\ark5_teacher_ep200_swinb_projector1376.pth.tar"
        else:
            self.src = rf"/home/hpc/iwi5/iwi5163h/checkpoints/ark5_teacher_ep200_swinb_projector1376.pth.tar"