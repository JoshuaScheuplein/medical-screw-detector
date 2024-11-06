# ------------------------------------------------------------------------------------
# Sparse DETR
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import math
from itertools import accumulate

import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from dataset.transforms import apply_transformation_matrix_batched, reverse_transformation_matrix_batched
from models.ops.functions import MSDeformAttnFunction
from utils.alpha_correspondance import project_to_second_view_fixed, project_to_second_view


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class MSAlphaCrossAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, embedding_size_vec=None):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level in the second view
        """

        assert n_levels == len(embedding_size_vec), "n_levels must be equal to the length of embedding_size_vec"

        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))

        self.im2col_step = 64
        self.python_ops_for_test = False

        self.d_model = d_model
        self.n_levels = n_levels

        self.n_heads = n_heads
        self.embedding_size_vec = embedding_size_vec
        self.embedding_cumsum = list(accumulate([0] + embedding_size_vec))
        self.embedding_size_sum = sum(embedding_size_vec)

        self.attention_weights = nn.Linear(d_model, n_heads * self.embedding_size_sum)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, p_pfws, pt_transforms, input_padding_mask=None):
        # """
        # :param query                       (N, Length_{query}, C)
        # :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
        #                                 or (N, Length_{query}, n_levels, 4),
        # :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        # :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        # :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        # :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        # :return output                     (N, Length_{query}, C)
        # """

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        # TODO
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.embedding_size_sum)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.embedding_size_sum)

        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2 and self.python_ops_for_test:

            # reverse transformation to apply epipolar geometry
            original_reference_points = reverse_transformation_matrix_batched(reference_points, pt_transforms) * 976.

            # compute alpha correspondance
            # recieve a line equation m * x + t
            ts, ms = project_to_second_view_fixed(p_pfws.to(torch.float32),
                                                  swap_consecutive_pairs(p_pfws).to(torch.float32),
                                                  original_reference_points)

            # Pre-allocate the output tensor
            reference_points_in_second_view = torch.empty([N, Len_q, self.embedding_size_sum, 2], device=ts.device)

            start_idx = 0
            for i, n_points in enumerate(self.embedding_size_vec):
                # Create xs once for all batches
                xs = torch.linspace(0, 976, n_points, device=ts.device).view(1, 1, -1, 1)

                # Perform the calculation
                end_idx = start_idx + n_points
                reference_points_in_second_view[:, :, start_idx:end_idx, :] = ts[:, :, i:i+1, :] + ms[:, :, i:i+1, :] * xs

                start_idx = end_idx

            reference_points_in_second_view = reference_points_in_second_view / 976.
            # TODO: fix outliers due to cropping
            transformed_reference_points = apply_transformation_matrix_batched(reference_points_in_second_view,
                                                                               swap_consecutive_pairs(pt_transforms))
            transformed_reference_points = transformed_reference_points.clamp(0, 1)
            transformed_reference_points = transformed_reference_points.view(N, Len_q, 1, self.embedding_size_sum, 2)

            # visualize(input_flatten, input_spatial_shapes, reference_points, transformed_reference_points, self.embedding_cumsum)

            sampling_locations = transformed_reference_points.repeat(1, 1, self.n_heads, 1, 1)

            output = ms_deform_attn_core_pytorch(swap_consecutive_pairs(value), input_spatial_shapes,
                                                     sampling_locations, attention_weights, self.embedding_cumsum)

        elif reference_points.shape[-1] == 2 and not self.python_ops_for_test:

            # reverse transformation to apply epipolar geometry
            original_reference_points = reverse_transformation_matrix_batched(reference_points, pt_transforms) * 976.

            # compute alpha correspondance
            # recieve a line equation m * x + t
            ts, ms = project_to_second_view_fixed(p_pfws.to(torch.float32),
                                                  swap_consecutive_pairs(p_pfws).to(torch.float32),
                                                  original_reference_points)

            # ts, ms = project_to_second_view(p_pfws.to(torch.float32),
            #                                       swap_consecutive_pairs(p_pfws).to(torch.float32),
            #                                       original_reference_points)

            output = torch.zeros(N, Len_q, self.d_model, device=value.device, dtype=value.dtype)

            start_idx = 0
            for i, n_points in enumerate(self.embedding_size_vec):
                xs = torch.linspace(0, 976, n_points, device=ts.device).view(1, 1, -1, 1)

                end_idx = start_idx + n_points
                reference_points_in_second_view = ts[:, :, i:i + 1, :] + ms[:, :, i:i + 1, :] * xs
                reference_points_in_second_view = reference_points_in_second_view / 976.
                # TODO: fix outliers due to cropping
                transformed_reference_points = apply_transformation_matrix_batched(reference_points_in_second_view, swap_consecutive_pairs(pt_transforms))
                transformed_reference_points = transformed_reference_points.clamp(0., 1.)
                transformed_reference_points = transformed_reference_points.view(N, Len_q, 1, n_points, 2)

                # visualize(input_flatten, input_spatial_shapes, reference_points, transformed_reference_points, self.embedding_cumsum)

                sampling_locations = transformed_reference_points.repeat(1, 1, self.n_heads, 1, 1)

                output += MSDeformAttnFunction.apply(swap_consecutive_pairs(value),
                                                     input_spatial_shapes[i:i + 1],
                                                     input_level_start_index[i:i + 1],
                                                     sampling_locations,
                                                     attention_weights[:, :, :, self.embedding_cumsum[i]:self.embedding_cumsum[i + 1]].contiguous(),
                                                     self.im2col_step)

                start_idx = end_idx

            output = output.contiguous()

        else:
            raise ValueError(
                'Last dim of reference_points must be 2, but get {} instead.'.format(reference_points.shape[-1]))

        output = self.output_proj(output)

        return output


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights, embedding_cumsum):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, P_sum_over_L, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_*M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, embedding_cumsum[lid_]:embedding_cumsum[lid_+1]].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(value_l_, sampling_grid_l_,
                                          mode='bilinear', padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_*M_, 1, Lq_, P_sum_over_L)
    output = (torch.cat(sampling_value_list, dim=-1) * attention_weights).sum(-1).view(N_, M_*D_, Lq_)
    return output.transpose(1, 2).contiguous()


def swap_consecutive_pairs(tensor):
    # Ensure the tensor has an even number of elements along the first dimension
    assert tensor.shape[0] % 2 == 0, "The first dimension must be even."

    # Reshape the tensor to group consecutive pairs
    reshaped = tensor.view(-1, 2, *tensor.shape[1:])

    # Swap the pairs
    swapped = reshaped.flip(1)

    # Reshape back to the original shape
    return swapped.reshape(tensor.shape)


def visualize(input_flatten, input_spatial_shapes, reference_points, sampling_locations, embedding_cumsum):
    # Split and reshape the tensor to extract the views
    views_multilevel = input_flatten.split([H * W for H, W in input_spatial_shapes], dim=1)

    # Define different colors for each point
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    for lid, (H, W) in enumerate(input_spatial_shapes[:1]):
        views = views_multilevel[lid][:2, :, :].view(2, H*W, -1).detach()
        view, orthogonal_view = views
        view -= view.mean(dim=0)
        orthogonal_view -= orthogonal_view.mean(dim=0)

        U_view, S_view, V_view = torch.pca_lowrank(view, q=3, center=True)
        # Project the data to the first 50 principal components
        view_pca = torch.matmul(view, V_view)
        view_pca = view_pca.view(H, W, 3).cpu().numpy()

        U_orthogonal, S_orthogonal, V_orthogonal = torch.pca_lowrank(orthogonal_view, q=3, center=True)
        # Project the data to the first 50 principal components
        orthogonal_pca = torch.matmul(orthogonal_view, V_orthogonal)
        orthogonal_pca = orthogonal_pca.view(H, W, 3).cpu().numpy()

        reference_points_view = reference_points[0, :, lid, :].detach().cpu().numpy()
        reference_points_in_orthogonal_view = sampling_locations[0, :, 0,
                                              embedding_cumsum[lid]:embedding_cumsum[lid + 1], :].cpu().numpy()

        for x in range(200):
            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Plot view
            ax1.imshow(view_pca, cmap='gray')
            for i in range(10):
                ax1.scatter(reference_points_view[i + x * 10, 0] * H.item(), reference_points_view[i + x * 10, 1] * W.item(),
                            color=colors[i], marker='x', s=100, linewidths=2)
            ax1.set_title("view")

            # Plot orthogonal view
            ax2.imshow(orthogonal_pca, cmap='gray')
            for i in range(10):
                ax2.scatter(reference_points_in_orthogonal_view[i + x * 10, ::20, 0] * H.item(), reference_points_in_orthogonal_view[i + x * 10, ::20, 1] * W.item(),
                            color=colors[i], marker='x', s=100, linewidths=2)
                ax2.plot([reference_points_in_orthogonal_view[i + x * 10, 0, 0] * H.item(), reference_points_in_orthogonal_view[i + x * 10, -1, 0] * H.item()],
                         [reference_points_in_orthogonal_view[i + x * 10, 0, 1] * W.item(), reference_points_in_orthogonal_view[i + x * 10, -1, 1] * W.item()],
                         color=colors[i], linestyle='--')
            ax2.set_title("orthogonal view")

            plt.title(f"View Layer {lid} for {x}`s 10 pts")
            plt.savefig(f"view_layer_{lid}_for_{x}s_10_pts.png")
