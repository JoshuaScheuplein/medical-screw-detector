import argparse

import numpy as np
import torch


class DefaultArgs:
    lr=0.0002
    lr_backbone_names=["backbone.0"]
    lr_backbone=0.00002
    lr_linear_proj_names=['reference_points', 'sampling_offsets']
    lr_linear_proj_mult=0.1
    batch_size=1
    weight_decay=0.0002
    epochs=50
    lr_drop=40
    lr_drop_epochs=None
    clip_max_norm=0.1

    # Variants of Deformable DETR
    with_box_refine=True
    two_stage=True

    # Model parameters
    frozen_weights=None

    # * Backbone
    backbone='resnet101'
    neglog_normalize=True
    position_embedding='sine'
    position_embedding_scale=2 * np.pi

    # * Transformer
    enc_layers=6
    dec_layers=6
    dim_feedforward=1024
    hidden_dim=256
    dropout=0.1
    nheads=8
    num_classes=1
    num_queries=20
    dec_n_points=4
    enc_n_points=4

    # * Efficient DETR
    eff_query_init=True
    eff_specific_head=True

    # * Sparse DETR
    use_enc_aux_loss=False
    rho=0.

    # * Alpha encoding
    alpha_correspondence=False

    # * Matcher
    set_cost_class = 5
    set_cost_screw_head_tip = 9.76
    set_cost_screw_midpoint = 0.976

    # * Loss coefficients
    cls_loss_coef = 5
    screw_head_tip_loss_coef = 0.01 # [0, 10]
    screw_midpoint_loss_coef = 0.001 # [0, 5]
    screw_angle_loss_coef = 0.001 # [0, 9]
    mask_prediction_coef = 1
    focal_alpha = -1

    # * dataset parameters
    data_dir = r'E:\MA_Data\V1-1to3objects-400projections-circular'
    dataset_reduction = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 4

    # * logging parameters
    log_wandb = False

    # * checkpointing
    result_dir = r"E:\MA_Data\results"

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector for Implants', add_help=False)
    parser.add_argument('--lr', default=DefaultArgs.lr, type=float)
    parser.add_argument('--lr_backbone_names', default=DefaultArgs.lr_backbone_names, type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=DefaultArgs.lr_backbone, type=float)
    parser.add_argument('--lr_linear_proj_names', default=DefaultArgs.lr_linear_proj_names, type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=DefaultArgs.lr_linear_proj_mult, type=float)
    parser.add_argument('--batch_size', default=DefaultArgs.batch_size, type=int)
    parser.add_argument('--weight_decay', default=DefaultArgs.weight_decay, type=float)
    parser.add_argument('--epochs', default=DefaultArgs.epochs, type=int)
    parser.add_argument('--lr_drop', default=DefaultArgs.lr_drop, type=int)
    parser.add_argument('--lr_drop_epochs', default=DefaultArgs.lr_drop_epochs, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=DefaultArgs.clip_max_norm, type=float,
                        help='gradient clipping max norm')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=DefaultArgs.with_box_refine, action='store_true')
    parser.add_argument('--two_stage', default=DefaultArgs.two_stage, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=DefaultArgs.frozen_weights,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default=DefaultArgs.backbone, type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--neglog_normalize', default=DefaultArgs.neglog_normalize, type=bool,
                        help="Normalize images with negative log before feeding them to the backbone")
    parser.add_argument('--position_embedding', default=DefaultArgs.position_embedding, type=str,
                        choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=DefaultArgs.position_embedding_scale, type=float,
                        help="position / size * scale")

    # * Transformer
    parser.add_argument('--enc_layers', default=DefaultArgs.enc_layers, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=DefaultArgs.dec_layers, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=DefaultArgs.dim_feedforward, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=DefaultArgs.hidden_dim, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=DefaultArgs.dropout, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=DefaultArgs.nheads, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_classes', default=DefaultArgs.num_classes, type=int,
                        help="Number of classes")
    parser.add_argument('--num_queries', default=DefaultArgs.num_queries, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=DefaultArgs.dec_n_points, type=int)
    parser.add_argument('--enc_n_points', default=DefaultArgs.enc_n_points, type=int)

    # * Efficient DETR
    parser.add_argument('--eff_query_init', default=DefaultArgs.eff_query_init, action='store_true')
    parser.add_argument('--eff_specific_head', default=DefaultArgs.eff_specific_head, action='store_true')

    # * Sparse DETR
    parser.add_argument('--use_enc_aux_loss', default=DefaultArgs.use_enc_aux_loss, action='store_true')
    parser.add_argument('--rho', default=DefaultArgs.rho, type=float)

    # * Alpha encoding
    parser.add_argument('--alpha_correspondence', default=DefaultArgs.alpha_correspondence, action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_true',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=DefaultArgs.set_cost_class, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_screw_head_tip', default=DefaultArgs.set_cost_screw_head_tip, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_screw_midpoint', default=DefaultArgs.set_cost_screw_midpoint, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=DefaultArgs.cls_loss_coef, type=float)
    parser.add_argument('--screw_head_tip_loss_coef', default=DefaultArgs.screw_head_tip_loss_coef, type=float)
    parser.add_argument('--screw_midpoint_loss_coef', default=DefaultArgs.screw_midpoint_loss_coef, type=float)
    parser.add_argument('--screw_angle_loss_coef', default=DefaultArgs.screw_angle_loss_coef, type=float)
    parser.add_argument('--mask_prediction_coef', default=DefaultArgs.mask_prediction_coef, type=float)
    parser.add_argument('--focal_alpha', default=DefaultArgs.focal_alpha, type=float)

    # * dataset parameters
    parser.add_argument('--data_dir', default=DefaultArgs.data_dir, type=str)
    parser.add_argument('--dataset_reduction', default=DefaultArgs.dataset_reduction, type=int)
    parser.add_argument('--device', default=DefaultArgs.device, help='device to use for training / testing')
    parser.add_argument('--num_workers', default=DefaultArgs.num_workers, type=int,
                        help='number of data loading workers')

    # * logging parameters
    parser.add_argument('--log_wandb', default=DefaultArgs.log_wandb, action='store_true')

    # * checkpointing
    parser.add_argument('--checkpoint_file', default=None, type=str,
                        help='path to checkpoint file')
    parser.add_argument('--result_dir', default=DefaultArgs.result_dir, type=str,
                        help='directory to store results like logs and checkpoints')

    return parser
