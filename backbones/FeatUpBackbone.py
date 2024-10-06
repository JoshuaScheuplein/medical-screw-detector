import numpy as np
import torch
import torchvision.transforms as T
from torch import Tensor
from torch.nn import functional

from backbones.BaseBackbone import BaseBackbone


class FeatUpBackbone(BaseBackbone):
    def __init__(self, args):
        if args.backbone.lower() == 'featup_dinov2_low_res':
            channels = [384, 384]
            image_size = 896
            embedding_size = [64, 128]

        elif args.backbone.lower() == 'featup_dinov2':
            channels = [384, 384, 384]
            image_size = 896
            embedding_size = [64, 128, 256]

        else:
            raise ValueError(f"FeatUpBackbone '{args.backbone}' not supported.")

        super().__init__(num_channels=channels, image_size=image_size, embedding_size=embedding_size, args=args)

    def instantiate(self):
        backbone_model = torch.hub.load("mhamilton723/FeatUp", "dinov2", use_norm=False)
        backbone_model.to(self.args.device)
        self.backbone = backbone_model

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
                T.ToTensor(),
                T.Resize(self.image_size, antialias=True),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            transformed_images.append(transform(img))

        img_batch = torch.stack(transformed_images)
        img_batch = img_batch.to("cuda")

        lr_features = self.backbone.model(img_batch)
        mid_features = self.upsample(lr_features, img_batch, self.backbone.upsampler.up1)
        if len(self.embedding_size) == 3:
            hr_features = self.upsample(mid_features, img_batch, self.backbone.upsampler.up2)

            # from head_tip_segmentation.scripts.check_sam import plot_pca_preview
            # plot_pca_preview(img_batch[0].permute(1, 2, 0), self.upsample(self.upsample(hr_features, img_batch, self.backbone.upsampler.up3), img_batch, self.backbone.upsampler.up4)[0].cpu().detach(), 384, 1024, show=True)
            # plot_pca_preview(img_batch[0].permute(1, 2, 0), lr_features[0].cpu().detach(), 384, 64, show=True)

            return [lr_features, mid_features, hr_features]
        return [lr_features, mid_features]

    def upsample(self, source, guidance, up):
        # taken from featUp
        _, _, h, w = source.shape
        small_guidance = functional.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled