import numpy as np
from torch import nn, Tensor


class BaseBackbone(nn.Module):

    def __init__(self, num_channels, image_size, embedding_size, args):
        super().__init__()
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_size = embedding_size
        self.args = args

    def get_backbone_name(self):
        return self.__class__.__name__ + self.variant

    def instantiate(self):
        raise NotImplementedError("")

    def forward(self, img_batch):
        raise NotImplementedError("")


