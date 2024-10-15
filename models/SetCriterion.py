import copy

import torch
from torch import nn
import torch.distributed as dist

from utils.dam import attn_map_to_flat_grid, idx_to_flat_grid, compute_corr
from utils.misc import sigmoid_focal_loss, accuracy, is_dist_avail_and_initialized, get_world_size
from utils.screw_ops import screw_head_tip_to_midpoint, \
    screw_cost_head, screw_cost_tip, screw_cost_angle, screw_cost_iou


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, no_object_category, matcher, weight_dict, losses, args):
        """ Create the criterion.
        Parameters:
            no_object_category: category id for no object
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.no_object_category = no_object_category
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses

        self.focal_alpha = args.focal_alpha
        self.eff_specific_head = args.eff_specific_head

    def loss_labels(self, outputs, targets, indices, num_screws, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        # extension for no object category
        target_classes = torch.full(src_logits.shape[:2], self.no_object_category,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # omit the last class (no object)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_screws, alpha=self.focal_alpha, gamma=2)
        loss_ce = loss_ce * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_precision'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_screws):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits > 0.).sum(1).squeeze(dim=1)
        card_err = nn.functional.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    @torch.no_grad()
    def loss_precision_recall(self, outputs, targets, indices, num_screws):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        head_tip_thresholds = torch.arange(1, 11, 1, device=outputs['pred_screws'].device).view(1, -1)
        midpoint_angle_thresholds = torch.arange(1, 11, 1, device=outputs['pred_screws'].device).view(1, -1)

        idx = self._get_src_permutation_idx(indices)
        src_screws = outputs['pred_screws'][idx]
        target_screws = torch.cat([t['screws'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        num_screws_in_targets = int(num_screws)
        num_screws_in_outputs = int((outputs['pred_logits'] > 0).sum().item())
        correctly_classified = outputs['pred_logits'][idx].squeeze(dim=1) > 0
        correctly_classified = correctly_classified.view(-1, 1)

        head_distances = torch.norm(src_screws[:, :2] - target_screws[:, :2], dim=1).view(-1, 1) * 976. * 0.305
        tip_distances = torch.norm(src_screws[:, 2:] - target_screws[:, 2:], dim=1).view(-1, 1) * 976. * 0.305

        midpoint_distances = torch.norm(screw_head_tip_to_midpoint(src_screws) - screw_head_tip_to_midpoint(target_screws), dim=1).view(-1, 1) * 976. * 0.305
        angle_distances = screw_cost_angle(src_screws, target_screws).view(-1, 1)

        for i, [x_head, y_head, x_tip, y_tip] in enumerate(target_screws):
            if ((x_head >= 1. or x_head <= 0. or y_head >= 1. or y_head <= 0.) or
                    (x_tip >= 1. or x_tip <= 0. or y_tip >= 1. or y_tip <= 0.)):
                head_distances[i] = 500
                tip_distances[i] = 500
                midpoint_distances[i] = 500
                angle_distances[i] = 500

                num_screws_in_outputs -= 1
                num_screws_in_targets -= 1

                # debug
                # print("Warning: Screw out of image bounds.")

        correct_head = head_distances < head_tip_thresholds
        TP_head = torch.sum(correctly_classified & correct_head, dim=0)
        precision_head = TP_head / num_screws_in_targets if num_screws_in_targets > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)
        recall_head = TP_head / num_screws_in_outputs if num_screws_in_outputs > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)

        correct_tip = tip_distances < head_tip_thresholds
        TP_tip = torch.sum(correctly_classified & correct_tip, dim=0)
        precision_tip = TP_tip / num_screws_in_targets if num_screws_in_targets > 0 else torch.zeros_like(TP_tip, device=TP_tip.device, dtype=torch.float32)
        recall_tip = TP_tip / num_screws_in_outputs if num_screws_in_outputs > 0 else torch.zeros_like(TP_tip, device=TP_tip.device, dtype=torch.float32)

        correct_midpoint = midpoint_distances < midpoint_angle_thresholds
        TP_midpoint = torch.sum(correctly_classified & correct_midpoint, dim=0)
        precision_midpoint = TP_midpoint / num_screws_in_targets if num_screws_in_targets > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)
        recall_midpoint = TP_midpoint / num_screws_in_outputs if num_screws_in_outputs > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)

        correct_angle = angle_distances < midpoint_angle_thresholds
        TP_angle = torch.sum(correctly_classified & correct_angle, dim=0)
        precision_angle = TP_angle / num_screws_in_targets if num_screws_in_targets > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)
        recall_angle = TP_angle / num_screws_in_outputs if num_screws_in_outputs > 0 else torch.zeros_like(TP_head, device=TP_head.device, dtype=torch.float32)

        losses = {}

        for i in range(10):
            losses[f"precision_head_{i+1}"] = precision_head[i]
            losses[f"recall_head_{i+1}"] = recall_head[i]
            losses[f"precision_tip_{i+1}"] = precision_tip[i]
            losses[f"recall_tip_{i+1}"] = recall_tip[i]
            losses[f"precision_midpoint_{i+1}"] = precision_midpoint[i]
            losses[f"recall_midpoint_{i+1}"] = recall_midpoint[i]
            losses[f"precision_angle_{i+1}"] = precision_angle[i]
            losses[f"recall_angle_{i+1}"] = recall_angle[i]

        return losses

    def loss_screws(self, outputs, targets, indices, num_screws):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_screws' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_screws = outputs['pred_screws'][idx]
        target_screws = torch.cat([t['screws'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        # L1 loss for screw midpoints
        loss_screw_midpoint = nn.functional.l1_loss(
            screw_head_tip_to_midpoint(src_screws * 976.),
            screw_head_tip_to_midpoint(target_screws * 976.),
            reduction='none')

        # L1 loss for screw angles with x-axis
        loss_screw_angle = screw_cost_angle(src_screws, target_screws)

        # L1 loss for screw head
        loss_screw_head = torch.diag(
            screw_cost_head(src_screws * 976., target_screws * 976.))

        # L1 loss for screw tip
        loss_screw_tip = torch.diag(
            screw_cost_tip(src_screws * 976., target_screws * 976.))

        for i, [x_head, y_head, x_tip, y_tip] in enumerate(target_screws):
            if ((x_head >= 1. and x_tip >= 1.) or (x_head <= 0. and x_tip <= 0.) or
                    (y_head >= 1. and y_tip >= 1.) or (y_head <= 0. and y_tip <= 0.)):
                loss_screw_head[i] = 0
                loss_screw_tip[i] = 0
                loss_screw_midpoint[i] = 0
                loss_screw_angle[i] = 0

                num_screws -= 1

        losses['loss_screw_midpoint'] = loss_screw_midpoint.sum() / num_screws
        losses['loss_screw_angle'] = loss_screw_angle.sum() / num_screws
        losses['loss_screw_head'] = loss_screw_head.sum() / num_screws
        losses['loss_screw_tip'] = loss_screw_tip.sum() / num_screws

        return losses

    def loss_mask_prediction(self, outputs, targets, indices, num_screws, layer=None):
        assert "backbone_mask_prediction" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        mask_prediction = outputs["backbone_mask_prediction"]
        loss_key = "loss_mask_prediction"

        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1, 2))

        losses = {}

        sparse_token_nums = outputs["sparse_token_nums"]
        num_topk = sparse_token_nums.max()

        topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
        target = torch.zeros_like(mask_prediction)
        for i in range(target.shape[0]):
            target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)

        losses.update({loss_key: nn.functional.multilabel_soft_margin_loss(mask_prediction, target)})

        return losses

    @torch.no_grad()
    def corr(self, outputs, targets, indices, num_screws):
        if "backbone_topk_proposals" not in outputs.keys():
            return {}

        assert "backbone_topk_proposals" in outputs
        assert "sampling_locations_dec" in outputs
        assert "attn_weights_dec" in outputs
        assert "spatial_shapes" in outputs
        assert "level_start_index" in outputs

        backbone_topk_proposals = outputs["backbone_topk_proposals"]
        sampling_locations_dec = outputs["sampling_locations_dec"]
        attn_weights_dec = outputs["attn_weights_dec"]
        spatial_shapes = outputs["spatial_shapes"]
        level_start_index = outputs["level_start_index"]

        flat_grid_topk = idx_to_flat_grid(spatial_shapes, backbone_topk_proposals)
        flat_grid_attn_map_dec = attn_map_to_flat_grid(
            spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1, 2))
        corr = compute_corr(flat_grid_topk, flat_grid_attn_map_dec, spatial_shapes)

        losses = {}
        losses["corr_mask_attn_map_dec_all"] = corr[0].mean()
        for i, _corr in enumerate(corr[1:]):
            losses[f"corr_mask_attn_map_dec_{i}"] = _corr.mean()
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_screws, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'screws': self.loss_screws,
            'precision': self.loss_precision_recall,
            "mask_prediction": self.loss_mask_prediction,
            "corr": self.corr,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_screws, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items()
                               if k not in ['aux_outputs', 'enc_outputs', 'backbone_outputs']}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        num_screws = sum(len(t["labels"]) for t in targets)
        num_screws = torch.as_tensor([num_screws], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            dist.all_reduce(num_screws)
        num_screws = torch.clamp(num_screws / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets, indices, num_screws, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['masks', "mask_prediction", "corr", "precision", "recall"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_screws, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            if not self.eff_specific_head:
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
            indices = self.matcher(enc_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr", "precision", "recall"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_screws, **kwargs)
                l_dict = {k + f'_enc': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'backbone_outputs' in outputs:
            backbone_outputs = outputs['backbone_outputs']
            bin_targets = copy.deepcopy(targets)
            if not self.eff_specific_head:
                for bt in bin_targets:
                    bt['labels'] = torch.zeros_like(bt['labels'])  # all labels are zero (meaning foreground)
            indices = self.matcher(backbone_outputs, bin_targets)
            for loss in self.losses:
                if loss in ['masks', "mask_prediction", "corr", "precision", "recall"]:
                    # Intermediate masks losses are too costly to compute, we ignore them.
                    continue
                kwargs = {}
                if loss == 'labels':
                    # Logging is enabled only for the last layer
                    kwargs['log'] = False
                l_dict = self.get_loss(loss, backbone_outputs, bin_targets, indices, num_screws, **kwargs)
                l_dict = {k + f'_backbone': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'aux_outputs_enc' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs_enc']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss in ['masks', "mask_prediction", "corr", "precision", "recall"]:
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_screws, **kwargs)
                    l_dict = {k + f'_enc_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses