import torch
from pytorch_lightning import LightningModule

from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from dataset.transforms import get_transformation_matrix
from models.DeformableDeTr import build_detr
from utils.misc import match_name_keywords


class DeformableDETRLightning(LightningModule):
    def __init__(self, args):
        super().__init__()
        detr_model, criterion = build_detr(args)
        self.detr_model = detr_model
        self.criterion = criterion
        self.args = args

    def training_step(self, batch, batch_idx):
        return self.generic_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self.generic_step(batch, mode="val")
    
    ########################################################################################################
    
    """ Original Code """
    def test_step(self, batch, batch_idx):
        return self.generic_step(batch, mode="test")

    """ This code was additionally added """
    # def test_epoch_end(self, outputs):
    #     # Option 2: Aggregate outputs manually if needed.
    #     # Here `outputs` is a list with the return of test_step for each batch.
    #     # For example, if you returned counts (TP, FP, FN), you can do:
    #     total_TP = sum(x["TP"] for x in outputs if "TP" in x)
    #     total_FP = sum(x["FP"] for x in outputs if "FP" in x)
    #     total_FN = sum(x["FN"] for x in outputs if "FN" in x)

    #     if total_TP + total_FP > 0:
    #         precision = total_TP / (total_TP + total_FP)
    #     else:
    #         precision = 0.0
    #     if total_TP + total_FN > 0:
    #         recall = total_TP / (total_TP + total_FN)
    #     else:
    #         recall = 0.0
    #     if precision + recall > 0:
    #         f1 = 2 * (precision * recall) / (precision + recall)
    #     else:
    #         f1 = 0.0

    #     # self.log("epoch_test_precision", precision)
    #     # self.log("epoch_test_recall", recall)
    #     # self.log("epoch_test_f1", f1)
    #     # You can also print them if you wish.
    #     print(f"Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    ########################################################################################################

    def generic_step(self, batch, mode):
        batch_size = self.args.batch_size

        projection, targets, indices = batch

        if self.args.alpha_correspondence:
            p_pfws = torch.stack([target["p_pfw"] for target in targets])
            pt_transform = torch.stack([
                get_transformation_matrix(target["h_flip"], target["v_flip"], target["rotation"], target["crop_region"], 976., p_pfws.device)
                for target in targets
            ])
        else:
            p_pfws = None
            pt_transform = None

        prediction = self.detr_model(projection, p_pfws, pt_transform)

        loss_dict = self.criterion(prediction, targets)

        for k, v in loss_dict.items():
            self.log(f"{mode}_{k}", v, on_epoch=True, on_step=True, batch_size=batch_size, sync_dist=True) # 'sync_dist=True' was additionally added

        weight_dict = self.criterion.weight_dict
        actual_loss_dict = {k: loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict}

        loss = sum(actual_loss_dict.values())
        self.log(f"{mode}_loss", loss, on_epoch=True, on_step=True, batch_size=batch_size, sync_dist=True) # 'sync_dist=True' was additionally added

        return {"loss": loss, "prediction": prediction}

    def configure_optimizers(self):
        lr_backbone_names = self.args.lr_backbone_names
        lr_linear_proj_names = self.args.lr_linear_proj_names
        lr = self.args.lr
        lr_backbone = self.args.lr_backbone
        lr_linear_proj_mult = self.args.lr_linear_proj_mult
        weight_decay = self.args.weight_decay
        lr_drop = self.args.lr_drop

        param_dicts = [
            {
                "params":
                    [p for n, p in self.named_parameters()
                     if (not match_name_keywords(n, lr_backbone_names)
                         and not match_name_keywords(n, lr_linear_proj_names)
                         and p.requires_grad)],
                "lr": lr,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if (match_name_keywords(n, lr_backbone_names)
                               and not match_name_keywords(n, lr_linear_proj_names)
                               and p.requires_grad)],
                "lr": lr_backbone,
            },
            {
                "params": [p for n, p in self.named_parameters()
                           if match_name_keywords(n, lr_linear_proj_names) and p.requires_grad],
                "lr": lr * lr_linear_proj_mult,
            }
        ]

        optimizer = AdamW(param_dicts, lr=lr, weight_decay=weight_decay)
        lr_scheduler = StepLR(optimizer, lr_drop)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }
