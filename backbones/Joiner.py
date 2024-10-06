import numpy as np
from torch import nn


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.num_channels = backbone.num_channels

    def forward(self, images: np.ndarray):
        # features
        out = self[0](images)

        # position encoding
        pos = []
        for x in out:
            pos.append(self[1](x).to(x.dtype))

        return out, pos
