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
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.screw_ops import screw_cost_mid, screw_head_tip_to_midpoint, screw_cost_head, screw_cost_tip


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cost_class: float = 1,
                 cost_screw_mid: float = 1,
                 cost_screw_head_tip: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_screw_mid = cost_screw_mid
        self.cost_screw_head_tip = cost_screw_head_tip
        assert cost_class != 0 or cost_screw_mid != 0 or cost_screw_head_tip != 0, "all costs cant be 0"

        ####################################################################################
        # Just for debugging and inspection ...
        # print("#######################################")
        # print("\nMATCHER:")
        # print(f"cost_class = {self.cost_class}")
        # print(f"cost_screw_mid = {self.cost_screw_mid}")
        # print(f"cost_screw_head_tip = {self.cost_screw_head_tip}")
        # print("#######################################")
        ####################################################################################

    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_screws": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        with torch.no_grad():
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
            out_screws = outputs["pred_screws"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_screws = torch.cat([v["screws"] for v in targets])

            ####################################################################################
            # Just for debugging and inspection ...
            # print("#######################################")
            # print("\nMATCHER:")
            # print(f"\ntgt_ids = {tgt_ids.shape}")
            # print(f"tgt_ids = {tgt_ids}")
            # print(f"\ntgt_screws = {tgt_screws.shape}")
            # print(f"tgt_screws = {tgt_screws}")
            # print("#######################################")
            ####################################################################################

            # Compute the classification cost.
            gamma = 2.0
            neg_cost_class = (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
            pos_cost_class = ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
            cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

            # Compute the L1 cost between screw midpoints
            cost_screw_mid = screw_cost_mid(
                screw_head_tip_to_midpoint(out_screws),
                screw_head_tip_to_midpoint(tgt_screws)
            )

            # Compute the L1? cost between boxes
            cost_screw_head_tip = screw_cost_head(out_screws, tgt_screws) + screw_cost_tip(out_screws, tgt_screws)

            # Final cost matrix
            C = self.cost_screw_mid * cost_screw_mid + self.cost_class * cost_class + self.cost_screw_head_tip * cost_screw_head_tip
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(v["screws"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor([_j % size for _j in j], dtype=torch.int64))
                    for (i, j), size in zip(indices, sizes)]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class,
                            cost_screw_mid=args.set_cost_screw_midpoint,
                            cost_screw_head_tip=args.set_cost_screw_head_tip)