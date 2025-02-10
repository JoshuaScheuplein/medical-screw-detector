############################
# # Default Args
############################

# # * Matcher
# set_cost_class = 5
# set_cost_screw_head_tip = 9.76
# set_cost_screw_midpoint = 0.976

# # * Loss coefficients
# cls_loss_coef = 5
# screw_head_tip_loss_coef = 0.01 # [0, 10]
# screw_midpoint_loss_coef = 0.001 # [0, 5]
# screw_angle_loss_coef = 0.001 # [0, 9]
# mask_prediction_coef = 1
# focal_alpha = -1

############################
# Job-924402
############################

# MATCHER: {cost_class = 5, cost_screw_mid = 0.976, cost_screw_head_tip = 9.76}

# weight_dict = {'loss_ce': 5, 'loss_screw_head': 0.01, 'loss_screw_tip': 0.01, 'loss_screw_midpoint': 0.001, 'loss_ce_enc': 5, 'loss_screw_head_enc': 0.01, 'loss_screw_tip_enc': 0.01, 'loss_screw_midpoint_enc': 0.001, 'loss_ce_enc_0': 5, 'loss_screw_head_enc_0': 0.01, 'loss_screw_tip_enc_0': 0.01, 'loss_screw_midpoint_enc_0': 0.001, 'loss_ce_enc_1': 5, 'loss_screw_head_enc_1': 0.01, 'loss_screw_tip_enc_1': 0.01, 'loss_screw_midpoint_enc_1': 0.001, 'loss_ce_enc_2': 5, 'loss_screw_head_enc_2': 0.01, 'loss_screw_tip_enc_2': 0.01, 'loss_screw_midpoint_enc_2': 0.001, 'loss_ce_enc_3': 5, 'loss_screw_head_enc_3': 0.01, 'loss_screw_tip_enc_3': 0.01, 'loss_screw_midpoint_enc_3': 0.001, 'loss_ce_enc_4': 5, 'loss_screw_head_enc_4': 0.01, 'loss_screw_tip_enc_4': 0.01, 'loss_screw_midpoint_enc_4': 0.001, 'loss_ce_backbone': 5, 'loss_screw_head_backbone': 0.01, 'loss_screw_tip_backbone': 0.01, 'loss_screw_midpoint_backbone': 0.001, 'loss_mask_prediction': 1}

############################
# Job-926383
############################

# MATCHER: {cost_class = 5, cost_screw_mid = 0.976, cost_screw_head_tip = 9.76}

# weight_dict = {'loss_ce': 5, 'loss_screw_head': 0.01, 'loss_screw_tip': 0.01, 'loss_screw_midpoint': 0.001, 'loss_ce_enc': 5, 'loss_screw_head_enc': 0.01, 'loss_screw_tip_enc': 0.01, 'loss_screw_midpoint_enc': 0.001, 'loss_ce_enc_0': 5, 'loss_screw_head_enc_0': 0.01, 'loss_screw_tip_enc_0': 0.01, 'loss_screw_midpoint_enc_0': 0.001, 'loss_ce_enc_1': 5, 'loss_screw_head_enc_1': 0.01, 'loss_screw_tip_enc_1': 0.01, 'loss_screw_midpoint_enc_1': 0.001, 'loss_ce_enc_2': 5, 'loss_screw_head_enc_2': 0.01, 'loss_screw_tip_enc_2': 0.01, 'loss_screw_midpoint_enc_2': 0.001, 'loss_ce_enc_3': 5, 'loss_screw_head_enc_3': 0.01, 'loss_screw_tip_enc_3': 0.01, 'loss_screw_midpoint_enc_3': 0.001, 'loss_ce_enc_4': 5, 'loss_screw_head_enc_4': 0.01, 'loss_screw_tip_enc_4': 0.01, 'loss_screw_midpoint_enc_4': 0.001, 'loss_ce_backbone': 5, 'loss_screw_head_backbone': 0.01, 'loss_screw_tip_backbone': 0.01, 'loss_screw_midpoint_backbone': 0.001, 'loss_mask_prediction': 1}

# """
# Modules to compute the matching cost and solve the corresponding LSAP.
# """
# import torch
# from torch import nn
# from scipy.optimize import linear_sum_assignment

# from utils.screw_ops import screw_cost_mid, screw_head_tip_to_midpoint, screw_cost_head, screw_cost_tip


# class HungarianMatcher(nn.Module):
#     """
#     This class computes an assignment between the targets and the predictions of the network.

#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """

#     def __init__(self,
#                  cost_class: float = 1,
#                  cost_screw_mid: float = 1,
#                  cost_screw_head_tip: float = 1):
#         """Creates the matcher

#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
#             cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
#         """
#         super().__init__()

#         self.cost_class = cost_class
#         self.cost_screw_mid = cost_screw_mid
#         self.cost_screw_head_tip = cost_screw_head_tip

#         assert cost_class != 0 or cost_screw_mid != 0 or cost_screw_head_tip != 0, "all costs cant be 0"

#     def forward(self, outputs, targets):
#         """
#         Performs the matching

#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_screws": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#         with torch.no_grad():
#             bs, num_queries = outputs["pred_logits"].shape[:2]

#             # We flatten to compute the cost matrices in a batch
#             out_prob = outputs["pred_logits"].flatten(0, 1).sigmoid()
#             out_screws = outputs["pred_screws"].flatten(0, 1)  # [batch_size * num_queries, 4]

#             # Also concat the target labels and boxes
#             tgt_ids = torch.cat([v["labels"] for v in targets])
#             tgt_screws = torch.cat([v["screws"] for v in targets])

#             # Compute the classification cost.
#             gamma = 2.0
#             neg_cost_class = (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
#             pos_cost_class = ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
#             cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

#             # Compute the L1 cost between screw midpoints
#             cost_screw_mid = screw_cost_mid(
#                 screw_head_tip_to_midpoint(out_screws),
#                 screw_head_tip_to_midpoint(tgt_screws)
#             )

#             # Compute the L1? cost between boxes
#             cost_screw_head_tip = screw_cost_head(out_screws, tgt_screws) + screw_cost_tip(out_screws, tgt_screws)

#             # Final cost matrix
#             C = self.cost_screw_mid * cost_screw_mid + self.cost_class * cost_class + self.cost_screw_head_tip * cost_screw_head_tip
#             C = C.view(bs, num_queries, -1).cpu()

#             sizes = [len(v["screws"]) for v in targets]
#             indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
#             return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor([_j % size for _j in j], dtype=torch.int64))
#                     for (i, j), size in zip(indices, sizes)]


# def build_matcher(args):
#     return HungarianMatcher(cost_class=args.set_cost_class,
#                             cost_screw_mid=args.set_cost_screw_midpoint,
#                             cost_screw_head_tip=args.set_cost_screw_head_tip)


import json
import argparse
from pathlib import Path

import numpy as np


# scan_names = ['Ankle18', 'Ankle19', 'Ankle20', 'Wrist08', 'Wrist09', 'Wrist10'] # Original code

scan_names = ['Ankle21', 'Ankle23', 'Elbow04', 'Wrist11', 'Wrist12', 'Wrist13', 'Spine06', 'Spine07', 'Ankle19', 'Wrist08', 'Wrist09', 'Wrist10'] # Adapted code


def compute_metrics(results_dir: Path):

    assert (results_dir.is_dir() == True) and (str(results_dir).split("/")[-1] == "V1-1to3objects-400projections-circular")

    diff, num_screws = 0, 0
    true_positives, false_positives, false_negatives = 0, 0, 0
    for scan in scan_names:
        for item in ["_le_512x512x512_1", "_le_512x512x512_2", "_le_512x512x512_3"]:
            sample = scan + item
            print("\n#####################################################")
            print(f"Processing sample '{sample}' ...")

            # Find and load predictions
            if (results_dir / Path(sample) / "predictions_test_50.json").is_file():
                file_path = results_dir / Path(sample) / "predictions_test_50.json"
            else:
                file_path = results_dir / Path(sample) / "predictions_val_49.json"
            assert file_path.is_file() == True, f"Missing prediction file for sample '{sample}' ..."
            with open(file_path, 'r') as f:
                sample_data = json.load(f)
                sample_data = sample_data["landmarks2d"]

            # Iterate through all views
            views = list(sample_data.keys())
            # assert len(views) == 33
            print(f"Found {len(views)} views in total for this sample: {views}")
            for view in views:
                predictions, targets = sample_data[view]["predictions"], sample_data[view]["targets"]
                num_predictions, num_targets = len(predictions.keys()), len(targets.keys())
                print(f"{view}: num_predictions = {num_predictions} & num_targets = {num_targets}")

                # Update number missed objects and total number of screws
                diff += np.abs(num_predictions - num_targets)
                num_screws += num_targets

                # if num_predictions > num_targets:
                #     true_positives += 



    print("\n#####################################################")
    print(f"Difference = {diff}")
    print(f"Number of screws = {num_screws}")
    print(f"Average Cardinality = {diff / num_screws}")
    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser('Computation of detection metrics for MICCAI paper')
    parser.add_argument('--results_dir', type=str)
    args = parser.parse_args()

    compute_metrics(Path(args.results_dir))
