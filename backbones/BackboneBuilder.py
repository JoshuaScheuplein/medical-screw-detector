from backbones.ARKBackbone import ARKBackbone
from backbones.DinoBackbone import DinoBackbone
from backbones.Joiner import Joiner
from backbones.MedicalResNet import MedicalResNet
from backbones.MedicalViT import MedicalViT
from backbones.PositionEncoding import build_position_encoding
from backbones.ResNetBackbone import ResNetBackbone
from backbones.SAMBackbone import SAMBackbone
from backbones.SAM2Backbone import SAM2Backbone


def build_backbone(args):
    backbone = get_backbone(args)
    backbone.instantiate()

    position_embedding = build_position_encoding(args)

    model = Joiner(backbone, position_embedding)

    return model


def get_backbone(args):
    if args.backbone in ["resnet50", "resnet101"]:
        backbone = ResNetBackbone(args)
    elif args.backbone == "sam_vit_b":
        backbone = SAMBackbone(args)
    elif args.backbone == "dinov2_vits14, dinov2_vitl14":
        backbone = DinoBackbone(args)
    elif args.backbone in ["ark5_low_res", "ark5", "ark5_single", "ark6_low_res", "ark6_very_low_res", "ark6", "ark6_single"]:
        backbone = ARKBackbone(args)
    elif args.backbone in ["sam2_base", "sam2_large"]:
        backbone = SAM2Backbone(args)
    elif args.backbone in ["medical_resnet50"]:
        backbone = MedicalResNet(args)
    elif args.backbone in ["medical_vit_s_8", "medical_vit_s_16"]:
        backbone = MedicalViT(args)

    else:
        raise ValueError(f"Invalid backbone type: {args.backbone}")
    return backbone
