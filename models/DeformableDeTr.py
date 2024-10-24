# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------------------


"""
Deformable DETR model and criterion classes.
"""
import numpy as np
import torch
from torch import nn
import math

from .DeformableTransformer import build_deforamble_transformer
from .MLP import MLP
from .Matcher import build_matcher
from .SetCriterion import SetCriterion
import copy

from backbones.BackboneBuilder import build_backbone
from dataset.Objects import ScrewEnum
from utils.misc import inverse_sigmoid


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DeformableDETR(nn.Module):
    """ This is the Deformable DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries,
                 aux_loss=True, with_box_refine=False, two_stage=False, args=None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. common/backbone/*.py
            transformer: torch module of the transformer architecture. See DeformableTransformer.py
            num_classes: number of object classes (exclusive NA)
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            with_box_refine: iterative bounding box refinement
            two_stage: two-stage Deformable DETR
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.screw_embed = MLP(hidden_dim, hidden_dim, output_dim=4, num_layers=3)
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
            # will be splited into query_embed(query_pos) & tgt later
        if len(backbone.num_channels) > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        self.use_enc_aux_loss = args.use_enc_aux_loss
        self.rho = args.rho

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        nn.init.constant_(self.screw_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.screw_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # hack implementation: a list of embedding heads (see the order)
        # n: dec_layers / m: enc_layers
        # [dec_0, dec_1, ..., dec_n-1, encoder, backbone, enc_0, enc_1, ..., enc_m-2]

        # at each layer of decoder (by default)
        num_pred = transformer.decoder.num_layers
        if self.two_stage:
            # at the end of encoder
            num_pred += 1
        if self.use_enc_aux_loss:
            # at each layer of encoder (excl. the last)
            num_pred += transformer.encoder.num_layers - 1

        if with_box_refine or self.use_enc_aux_loss:
            # individual heads with the same initialization
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.screw_embed = _get_clones(self.screw_embed, num_pred)
            nn.init.constant_(self.screw_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            # shared heads
            nn.init.constant_(self.screw_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.screw_embed = nn.ModuleList([self.screw_embed for _ in range(num_pred)])

        if two_stage:
            # hack implementation
            self.transformer.decoder.class_embed = self.class_embed
            self.transformer.decoder.screw_embed = self.screw_embed
            for box_embed in self.transformer.decoder.screw_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

        if self.use_enc_aux_loss:
            # the output from the last layer should be specially treated as an input of decoder
            num_layers_excluding_the_last = transformer.encoder.num_layers - 1
            self.transformer.encoder.aux_heads = True
            self.transformer.encoder.class_embed = self.class_embed[-num_layers_excluding_the_last:]
            self.transformer.encoder.screw_embed = self.screw_embed[-num_layers_excluding_the_last:]
            for box_embed in self.transformer.encoder.screw_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def forward(self, samples: np.ndarray, p_pfws: torch.Tensor, pt_transform):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - p_pfws: projection matrix of detector, of shape [batch_size x 3 x 4]
               - pt_transform: PointTransforms containing transform information of images, of shape [batch_size x num_queries]

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_screws": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        ###########
        # Backbone

        features, pos = self.backbone(samples)

        srcs = []
        for l, feature in enumerate(features):
            srcs.append(self.input_proj[l](feature))

            # # debug
            # from tifffile import imwrite
            # imwrite(f"feat_{l}.tiff", feature[0].detach().cpu().numpy(), dtype=np.float32)
            # imwrite(f"src_{l}.tiff", self.input_proj[l](feature)[0].detach().cpu().numpy(), dtype=np.float32)

        #
        # # multi-scale features smaller than C5 projected with 2 strided 3x3 conv
        # if self.num_feature_levels > len(srcs):
        #     _len_srcs = len(srcs)
        #     for l in range(_len_srcs, self.num_feature_levels):
        #         if l == _len_srcs:
        #             # feature scale 1/32
        #             src = self.input_proj[l](features[-1].tensors)
        #         else:
        #             # feature scale <1/64: recursively downsample the last projection
        #             src = self.input_proj[l](srcs[-1])
        #         m = samples.mask
        #         mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
        #         pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
        #         srcs.append(src)
        #         masks.append(mask)
        #         pos.append(pos_l)

        ###########
        # Transformer encoder & decoder
        query_embeds = None
        if not self.two_stage:
            query_embeds = self.query_embed.weight
        (hs, init_reference, inter_references,
         enc_outputs_class, enc_outputs_coord_unact,
         backbone_mask_prediction,
         enc_inter_outputs_class, enc_inter_outputs_coord,
         sampling_locations_enc, attn_weights_enc,
         sampling_locations_dec, attn_weights_dec,
         backbone_topk_proposals, spatial_shapes, level_start_index) = \
            self.transformer(srcs, pos, p_pfws, pt_transform, query_embeds)

        ###########
        # Detection heads
        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(hs)):
            # lvl: level of decoding layer
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_coord = self.screw_embed[lvl](hs[lvl])

            assert init_reference is not None and inter_references is not None
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            if reference.shape[-1] == 4:
                outputs_coord += reference
            else:
                assert reference.shape[-1] == 2
                outputs_coord[..., :2] += reference

            outputs_coord = outputs_coord.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        # the topmost layer output
        out = {
            "pred_logits": outputs_class[-1],
            "pred_screws": outputs_coord[-1],
            "sampling_locations_enc": sampling_locations_enc,
            "attn_weights_enc": attn_weights_enc,
            "sampling_locations_dec": sampling_locations_dec,
            "attn_weights_dec": attn_weights_dec,
            "spatial_shapes": spatial_shapes,
            "level_start_index": level_start_index,
        }
        if backbone_topk_proposals is not None:
            out["backbone_topk_proposals"] = backbone_topk_proposals

        if self.aux_loss:
            # make loss from every intermediate layers (excluding the last one)
            out['aux_outputs'] = self._set_aux_loss(outputs_class[:-1], outputs_coord[:-1])

        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_screws': enc_outputs_coord}

        if self.rho:
            out["backbone_mask_prediction"] = backbone_mask_prediction

        if self.use_enc_aux_loss:
            out['aux_outputs_enc'] = self._set_aux_loss(enc_inter_outputs_class, enc_inter_outputs_coord)

        if self.rho:
            out["sparse_token_nums"] = self.transformer.sparse_token_nums

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_screws': b}
                for a, b in zip(outputs_class, outputs_coord)]


def build_detr(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    args.backbone = backbone
    transformer = build_deforamble_transformer(args)
    total_parameters = 0
    for parameter in backbone.parameters():
        total_parameters += parameter.numel()
    print(f"\nTotal Backbone Parameters: {total_parameters}\n")

    model = DeformableDETR(
        backbone,
        transformer,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        args=args,
    )

    matcher = build_matcher(args)
    weight_dict = {
        'loss_ce': args.cls_loss_coef,
        'loss_screw_head': args.screw_head_tip_loss_coef,
        'loss_screw_tip': args.screw_head_tip_loss_coef,
        'loss_screw_midpoint':  args.screw_midpoint_loss_coef
    }

    # TODO this is a hack
    aux_weight_dict = {}

    if args.aux_loss:
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})

    if args.two_stage:
        aux_weight_dict.update({k + f'_enc': v for k, v in weight_dict.items()})

    if args.use_enc_aux_loss:
        for i in range(args.enc_layers - 1):
            aux_weight_dict.update({k + f'_enc_{i}': v for k, v in weight_dict.items()})

    if args.rho:
        aux_weight_dict.update({k + f'_backbone': v for k, v in weight_dict.items()})

    if aux_weight_dict:
        weight_dict.update(aux_weight_dict)

    weight_dict['loss_mask_prediction'] = args.mask_prediction_coef

    losses = ['labels', 'screws', 'cardinality', "corr", "precision"]

    if args.rho:
        losses += ["mask_prediction"]

    # num_classes, matcher, weight_dict, losses, focal_alpha=0.25
    criterion = SetCriterion(ScrewEnum.NA, matcher, weight_dict, losses, args)
    criterion.to(device)

    return model, criterion
