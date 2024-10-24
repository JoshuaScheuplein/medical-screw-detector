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


import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from models.ops.modules.ms_alpha_cross_attn import MSAlphaCrossAttn
from utils.misc import inverse_sigmoid
from models.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=False,
                 num_feature_levels=4, dec_n_points=4, enc_n_points=4,
                 two_stage=False, two_stage_num_proposals=300,
                 args=None):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.eff_query_init = args.eff_query_init
        self.eff_specific_head = args.eff_specific_head
        # there's no need to compute reference points if above 2 conditions meet simultaneously
        self._log_args('eff_query_init', 'eff_specific_head')

        self.rho = args.rho
        self.use_enc_aux_loss = args.use_enc_aux_loss
        self.sparse_enc_head = 1 if self.two_stage and self.rho else 0

        if self.rho:
            self.enc_mask_predictor = MaskPredictor(self.d_model, self.d_model)
        else:
            self.enc_mask_predictor = None

        encoder_layer = DeformableTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation,
                                                          num_feature_levels, nhead, enc_n_points, args.backbone[0].embedding_size)
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, self.d_model)

        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                          dropout, activation,
                                                          num_feature_levels, nhead, dec_n_points)
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if self.two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

        if self.two_stage:
            self.pos_trans = nn.Linear(d_model * 2, d_model * (1 if self.eff_query_init else 2))
            self.pos_trans_norm = nn.LayerNorm(d_model * (1 if self.eff_query_init else 2))

        if not self.two_stage:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _log_args(self, *names):
        print('=======================')
        print("\n".join([f"{name}: {getattr(self, name)}" for name in names]))
        print('=======================')

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if hasattr(self, 'reference_points'):
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        # proposals: N, L(top_k), 4(bbox coords.)
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)  # 128
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        proposals = proposals.sigmoid() * scale  # N, L, 4
        pos = proposals[:, :, :, None] / dim_t  # N, L, 4, 128
        # apply sin/cos alternatively
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4)  # N, L, 4, 64, 2
        pos = pos.flatten(2)  # N, L, 512 (4 x 128)
        return pos

    def gen_encoder_output_proposals(self, memory, spatial_shapes, process_output=True):
        """Make region proposals for each multi-scale features considering their shapes and padding masks,
        and project & normalize the encoder outputs corresponding to these proposals.
            - center points: relative grid coordinates in the range of [0.01, 0.99] (additional mask)
            - width/height:  2^(layer_id) * s (s=0.05) / see the appendix A.4

        Tensor shape example:
            Args:
                memory: torch.Size([2, 15060, 256])
                memory_padding_mask: torch.Size([2, 15060])
                spatial_shape: torch.Size([4, 2])
            Returns:
                output_memory: torch.Size([2, 15060, 256])
                    - same shape with memory ( + additional mask + linear layer + layer norm )
                output_proposals: torch.Size([2, 15060, 4])
                    - x, y, w, h
        """
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            # level of encoded feature scale
            # mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            # valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            # valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                                            indexing='ij')
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.tensor([W_, H_], device=grid.device).view(1, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            # TODO: MAX wrong prediction (maybe reverse grid)
            # reverse_grid = torch.cat([torch.flipgrid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)

        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))  # inverse of sigmoid
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))  # sigmoid(inf) = 1

        output_memory = memory
        if process_output:
            output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, torch.full((N_, 1), S_)

    def forward(self, srcs, pos_embeds, p_pfws=None, pt_transform=None, query_embed=None):
        assert self.two_stage or query_embed is not None

        ###########
        # prepare input for encoder
        src_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        # valid ratios across multi-scale features of the same image can be varied,
        # while they are interpolated and binarized on different resolutions.
        # valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        ###########
        # prepare for sparse encoder
        if self.rho or self.use_enc_aux_loss:
            backbone_output_memory, backbone_output_proposals, valid_token_nums = self.gen_encoder_output_proposals(
                src_flatten + lvl_pos_embed_flatten, spatial_shapes,
                process_output=bool(self.rho))
            self.valid_token_nums = valid_token_nums

        if self.rho:
            sparse_token_nums = (valid_token_nums * self.rho).int() + 1
            backbone_topk = int(max(sparse_token_nums))
            self.sparse_token_nums = sparse_token_nums

            backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])

            backbone_mask_prediction = self.enc_mask_predictor(backbone_output_memory).squeeze(-1)

            backbone_topk_proposals = torch.topk(backbone_mask_prediction, backbone_topk, dim=1)[1]
        else:
            backbone_topk_proposals = None
            sparse_token_nums = None

        ###########
        # encoder
        if self.encoder:
            output_proposals = backbone_output_proposals if self.use_enc_aux_loss else None
            encoder_output = self.encoder(src_flatten, spatial_shapes, level_start_index,
                                          p_pfws=p_pfws, pt_transform=pt_transform,
                                          pos=lvl_pos_embed_flatten,
                                          topk_inds=backbone_topk_proposals, output_proposals=output_proposals,
                                          sparse_token_nums=sparse_token_nums)

            memory, sampling_locations_enc, attn_weights_enc = encoder_output[:3]

            if self.use_enc_aux_loss:
                enc_inter_outputs_class, enc_inter_outputs_coord_unact = encoder_output[3:5]
        else:
            memory = src_flatten + lvl_pos_embed_flatten

        ###########
        # prepare input for decoder
        bs, _, c = memory.shape  # torch.Size([N, L, 256])
        topk_proposals = None
        if self.two_stage:
            # finalize the first stage output
            # project & normalize the memory and make proposal bounding boxes on them
            output_memory, output_proposals, _ = self.gen_encoder_output_proposals(memory, spatial_shapes)

            # hack implementation for two-stage Deformable DETR (using the last layer registered in class/screw_embed)
            # 1) a linear projection for bounding box binary classification (fore/background)
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            # 2) 3-layer FFN for bounding box regression
            enc_outputs_coord_offset = self.decoder.screw_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = output_proposals + enc_outputs_coord_offset  # appendix A.4

            # top scoring bounding boxes are picked as the final region proposals.
            # these proposals are fed into the decoder as initial boxes for the iterative bounding box refinement.
            topk = self.two_stage_num_proposals
            # enc_outputs_class: torch.Size([N, L, 91])

            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[..., 0]
            else:
                # take the score from the binary(fore/background) classfier
                # though outputs have 91 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()

            init_reference_out = reference_points
            # pos_embed -> linear layer -> layer norm
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))

            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                tgt = torch.gather(memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, memory.size(-1)))
                query_embed = pos_trans_out
            else:
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)

        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points

        ###########
        # decoder
        hs, inter_references, sampling_locations_dec, attn_weights_dec = self.decoder(tgt, reference_points, src=memory,
                                                                                      src_spatial_shapes=spatial_shapes,
                                                                                      src_level_start_index=level_start_index,
                                                                                      query_pos=query_embed,
                                                                                      topk_inds=topk_proposals)

        inter_references_out = inter_references

        ret = []
        ret += [hs, init_reference_out, inter_references_out]
        ret += [enc_outputs_class, enc_outputs_coord_unact] if self.two_stage else [None] * 2
        if self.rho:
            ret += [backbone_mask_prediction]
        else:
            ret += [None]
        ret += [enc_inter_outputs_class, enc_inter_outputs_coord_unact] if self.use_enc_aux_loss else [None] * 2
        ret += [sampling_locations_enc, attn_weights_enc, sampling_locations_dec, attn_weights_dec]
        ret += [backbone_topk_proposals, spatial_shapes, level_start_index]
        return ret


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, embedding_size_vec=None):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.cross_attn = MSAlphaCrossAttn(d_model, n_levels, n_heads, embedding_size_vec)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, p_pfws=None, pt_transform=None, tgt=None):
        if tgt is None:
            # self attention
            src2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(src, pos),
                                                                    reference_points, src, spatial_shapes,
                                                                    level_start_index)
            src = src + self.dropout1(src2)
            src = self.norm1(src)
            # torch.Size([2, 13101, 256])

            if p_pfws is not None and pt_transform is not None:
                # TODO: cross attn
                src2 = self.cross_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes,
                                       level_start_index, p_pfws, pt_transform)
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                # torch.Size([2, 13101, 256])

            # ffn
            src = self.forward_ffn(src)

            return src, sampling_locations, attn_weights
        else:
            # self attention
            tgt2, sampling_locations, attn_weights = self.self_attn(self.with_pos_embed(tgt, pos),
                                                                    reference_points, src, spatial_shapes,
                                                                    level_start_index)
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

            if p_pfws is not None and pt_transform is not None:
                # TODO: cross attn
                tgt2 = self.cross_attn(self.with_pos_embed(tgt, pos), reference_points, src, spatial_shapes,
                                       level_start_index, p_pfws, pt_transform)
                tgt = tgt + self.dropout1(tgt2)
                tgt = self.norm1(tgt)
                # torch.Size([2, 13101, 256])

            # ffn
            tgt = self.forward_ffn(tgt)

            return tgt, sampling_locations, attn_weights


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, mask_predictor_dim=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # hack implementation
        self.aux_heads = False
        self.class_embed = None
        self.screw_embed = None

    @staticmethod
    def get_reference_points(spatial_shapes, device):
        """Make reference points for every single point on the multi-scale feature maps.
        Each point has K reference points on every the multi-scale features.
        """
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                                          indexing='ij')
            ref_y = ref_y.reshape(-1)[None] / H_
            ref_x = ref_x.reshape(-1)[None] / W_
            # out-of-reference points have relative coords. larger than 1
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None]

        return reference_points

    # TODO: MAX valid_ratios - looks good
    def forward(self, src, spatial_shapes, level_start_index, p_pfws=None, pt_transform=None,
                pos=None, topk_inds=None, output_proposals=None, sparse_token_nums=None):
        if self.aux_heads:
            assert output_proposals is not None
        else:
            assert output_proposals is None

        output = src
        sparsified_keys = False if topk_inds is None else True
        reference_points = self.get_reference_points(spatial_shapes, device=src.device)
        reference_points = reference_points.repeat(src.shape[0], 1, spatial_shapes.shape[0], 1)

        sampling_locations_all = []
        attn_weights_all = []
        if self.aux_heads:
            enc_inter_outputs_class = []
            enc_inter_outputs_coords = []

        if sparsified_keys:
            assert topk_inds is not None
            B_, N_, S_, P_ = reference_points.shape
            reference_points = torch.gather(reference_points.view(B_, N_, -1), 1,
                                            topk_inds.unsqueeze(-1).repeat(1, 1, S_ * P_)).view(B_, -1, S_, P_)
            tgt = torch.gather(output, 1, topk_inds.unsqueeze(-1).repeat(1, 1, output.size(-1)))
            pos = torch.gather(pos, 1, topk_inds.unsqueeze(-1).repeat(1, 1, pos.size(-1)))
            if output_proposals is not None:
                output_proposals = output_proposals.gather(1, topk_inds.unsqueeze(-1).repeat(1, 1,
                                                                                             output_proposals.size(-1)))
        else:
            tgt = None

        for lid, layer in enumerate(self.layers):
            # if tgt is None: self-attention / if tgt is not None: cross-attention w.r.t. the target queries
            tgt, sampling_locations, attn_weights = layer(output, pos, reference_points, spatial_shapes,
                                                          level_start_index, p_pfws, pt_transform,
                                                          tgt=tgt if sparsified_keys else None)
            sampling_locations_all.append(sampling_locations)
            attn_weights_all.append(attn_weights)
            if sparsified_keys:
                if sparse_token_nums is None:
                    output = output.scatter(1, topk_inds.unsqueeze(-1).repeat(1, 1, tgt.size(-1)), tgt)
                else:
                    outputs = []
                    for i in range(topk_inds.shape[0]):
                        outputs.append(
                            output[i].scatter(0,
                                              topk_inds[i][:sparse_token_nums[i]].unsqueeze(-1).repeat(1, tgt.size(-1)),
                                              tgt[i][:sparse_token_nums[i]])
                        )
                    output = torch.stack(outputs)
            else:
                output = tgt

            if self.aux_heads and lid < self.num_layers - 1:
                # feed outputs to aux. heads
                output_class = self.class_embed[lid](tgt)
                output_offset = self.screw_embed[lid](tgt)
                output_coords_unact = output_proposals + output_offset
                # values to be used for loss compuation
                enc_inter_outputs_class.append(output_class)
                enc_inter_outputs_coords.append(output_coords_unact.sigmoid())

        # Change dimension from [num_layer, batch_size, ...] to [batch_size, num_layer, ...]
        sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        attn_weights_all = torch.stack(attn_weights_all, dim=1)

        ret = [output, sampling_locations_all, attn_weights_all]

        if self.aux_heads:
            ret += [enc_inter_outputs_class, enc_inter_outputs_coords]

        return ret


class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes,
                level_start_index):
        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # cross attention
        assert reference_points is not None, "deformable attention needs reference points!"
        tgt2, sampling_locations, attn_weights = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                                                 reference_points,
                                                                 src, src_spatial_shapes, level_start_index)

        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)
        # torch.Size([2, 300, 256])

        return tgt, sampling_locations, attn_weights


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.screw_embed = None
        self.class_embed = None

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, query_pos=None,
                topk_inds=None):
        """
        Args:
            tgt: torch.Size([2, 300, 256]) (query vectors)
            reference_points: torch.Size([2, 300, 2])
            src: torch.Size([2, 13101, 256]) (last MS feature map from the encoder)
            query_pos: torch.Size([2, 300, 256]) (learned positional embedding of query vectors)
            - `tgt` and `query_pos` are originated from the same query embedding.
            - `tgt` changes through the forward pass as object query vector
               while `query_pos` does not and is added as positional embedding.

        Returns: (when return_intermediate=True)
            output: torch.Size([6, 2, 300, 256])
            reference_points: torch.Size([6, 2, 300, 2])
        """
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        sampling_locations_all = []
        attn_weights_all = []
        for lid, layer in enumerate(self.layers):

            if reference_points is None:
                reference_points_input = None
            elif reference_points.shape[-1] == 4:
                # output from iterative bounding box refinement
                # reference_points: N, top_k, 4(x1/y1/x2/y2)
                # src_valid_ratios: N, num_feature_levels, 2(w/h) of img
                # reference_points_input: N, top_k, num_feature_levels, 4(x1/y1/x2/y2)
                # TODO: MAX check reference_points - looks good (src_valid_ratios should always be in [0,1])
                reference_points_input = reference_points[:, :, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None]

            output, sampling_locations, attn_weights = layer(output, query_pos, reference_points_input, src,
                                                             src_spatial_shapes,
                                                             src_level_start_index)
            sampling_locations_all.append(sampling_locations)
            attn_weights_all.append(attn_weights)

            # hack implementation for iterative bounding box refinement
            if self.screw_embed is not None:
                assert reference_points is not None, "box refinement needs reference points!"
                tmp = self.screw_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    # TODO: MAX check reference_points - looks good
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        # Change dimension from [num_layer, batch_size, ...] to [batch_size, num_layer, ...]
        sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        attn_weights_all = torch.stack(attn_weights_all, dim=1)

        if self.return_intermediate:
            intermediate_outputs = torch.stack(intermediate)
            if intermediate_reference_points[0] is None:
                intermediate_reference_points = None
            else:
                intermediate_reference_points = torch.stack(intermediate_reference_points)

            return intermediate_outputs, intermediate_reference_points, sampling_locations_all, attn_weights_all

        return output, reference_points, sampling_locations_all, attn_weights_all


class MaskPredictor(nn.Module):
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )

    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=len(args.backbone.num_channels),
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=args.two_stage,
        two_stage_num_proposals=args.num_queries,
        args=args)
