import numpy as np
import torch
from torch import Tensor
from torchvision.models import resnet50, resnet101
import torchvision.transforms as T
from torchvision.models._utils import IntermediateLayerGetter

from backbones.BaseBackbone import BaseBackbone


class ResNetBackbone(BaseBackbone):
    def __init__(self, args):

        image_size = 1024

        if args.backbone == 'resnet50':
            channels = [512, 1024, 2048]
            embedding_size = [128, 64, 32]
        elif args.backbone == 'resnet50_single':
            channels = [1024]
            embedding_size = [64]
        elif args.backbone == 'resnet101':
            channels = [512, 1024, 2048]
            embedding_size = [128, 64, 32]
        elif args.backbone == 'resnet101_single':
            channels = [1024]
            embedding_size = [64]
        else:
            raise ValueError(f"ResNetBackbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        if self.args.backbone == 'resnet50' or self.args.backbone == 'resnet50_single':
            self.backbone = resnet50(weights='DEFAULT').cuda()
            self.ilg = IntermediateLayerGetter(self.backbone, {'layer2': '2', 'layer3': '3', 'layer4': '4'})

        elif self.args.backbone == 'resnet101' or self.args.backbone == 'resnet101_single':
            self.backbone = resnet101(weights='DEFAULT').cuda()
            self.ilg = IntermediateLayerGetter(self.backbone, {'layer2': '2', 'layer3': '3', 'layer4': '4'})

    def forward(self, img_batch):

        transformed_images = []

        for img in img_batch:
            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            transform = T.Compose([
                T.ToTensor(),
                lambda x: 255.0 * x,  # scale by 255
            ])

            transformed_images.append(transform(img))

        img_batch = torch.stack(transformed_images)
        img_batch = img_batch.to(self.args.device)

        layers = self.ilg(img_batch)

        return list(layers.values())