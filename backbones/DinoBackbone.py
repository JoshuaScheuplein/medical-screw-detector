import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor

from backbones.BaseBackbone import BaseBackbone


class DinoBackbone(BaseBackbone):
    def __init__(self, args):
        # Implement logic to select and instantiate the chosen backbone
        if args.backbone.lower() == "dinov2_vits14":
            channels = [384]
            image_size = 896
            embedding_size = [64]

        elif args.backbone.lower() == "dinov2_vitl14":
            channels = [1024]
            image_size = 896
            embedding_size = [64]

        else:
            raise ValueError(f"SAM Backbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.args.backbone.lower())
        backbone_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.backbone = backbone_model

    '''
        Input image batch of shape (B, C, H, W) 
        Return the embeddings of the image as list of tensors of shape (B, num_channel, embedding_size, embedding_size)
    '''
    def forward(self, img_batch):

        transformed_images = []

        for img in img_batch:

            img = np.expand_dims(img, axis=0)
            img = np.repeat(img, 3, axis=0)
            img = np.transpose(img, (1, 2, 0))

            transform = T.Compose([
                T.ToTensor(),
                lambda x: 255.0 * x,  # scale by 255
                T.Normalize(
                    mean=(123.675, 116.28, 103.53),
                    std=(58.395, 57.12, 57.375)
                )
            ])

            transformed_images.append(transform(img))

        img_batch = torch.stack(transformed_images)
        img_batch = img_batch.to("cuda")
        img_features = self.backbone.get_intermediate_layers(img_batch)[0]

        img_features = img_features.view(img_features.size(0), self.embedding_size[0], self.embedding_size[0], self.num_channels[0])
        img_features = img_features.permute(0, 3, 1, 2)

        return [img_features]



