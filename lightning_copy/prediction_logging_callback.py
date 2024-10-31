import json
import os
import warnings
from typing import Any, Optional

from pytorch_lightning import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT


class PredictionLoggingCallback(Callback):

    def __init__(self, result_dir, batch_size: int = 1):
        super().__init__()
        self.labels_list = []
        self.result_dir = result_dir

        self.log_index = 40 // batch_size

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.create_empty_labels_list(trainer.val_dataloaders.dataset)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.create_empty_labels_list(trainer.test_dataloaders.dataset)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.dump_labels_list(trainer.val_dataloaders.dataset, trainer.current_epoch, "val")

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.dump_labels_list(trainer.test_dataloaders.dataset, trainer.current_epoch, "test")

    def on_validation_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: Optional[STEP_OUTPUT],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        if batch_idx % self.log_index == 0:
            dataloader = trainer.val_dataloaders
            self.write_result(dataloader, batch, outputs)

    def on_test_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            **kwargs
    ) -> None:
        dataloader = trainer.test_dataloaders
        self.write_result(dataloader, batch, outputs)

    def create_empty_labels_list(self, dataset):
        self.labels_list = []
        for i in range(len(dataset.volume_names)):
            self.labels_list.append(
                {
                    "landmarks2d": {}
                }
            )

    def dump_labels_list(self, dataset, epoch, state):
        for volume_idx in range(len(self.labels_list)):
            prediction_file_path = os.path.join(self.result_dir,
                                                os.path.basename(os.path.normpath(dataset.data_dir)),
                                                dataset.volume_names[volume_idx],
                                                f"predictions_{state}_{epoch}.json")

            if os.path.exists(prediction_file_path):
                warnings.warn(f"Warning: Overwriting existing prediction file {prediction_file_path}.")

            with open(prediction_file_path, 'w') as fp:
                json.dump(self.labels_list[volume_idx], fp=fp, indent=4)

    def write_result(self, data_loader, inputs, outputs):
        _, target, indices = inputs
        index = indices[0]

        target_0 = target[0]

        coords_tgt = target_0["screws"]

        rotation = target_0["rotation"] if "rotation" in target_0 else 0
        crop_region = target_0["crop_region"] if "crop_region" in target_0 else [0, 0, 976, 976]
        v_flip = target_0["v_flip"] if "v_flip" in target_0 else False
        h_flip = target_0["h_flip"] if "h_flip" in target_0 else False

        coords_pred = outputs["prediction"]["pred_screws"][0]
        logits_pred = outputs["prediction"]["pred_logits"][0]

        volume_idx = index // data_loader.dataset.images_per_volume
        view_idx = index % data_loader.dataset.images_per_volume

        labels: dict = self.labels_list[volume_idx]

        labels["landmarks2d"][f"view_{view_idx}"] = {
            "rotation": rotation,
            "crop_region": crop_region,
            "v_flip": v_flip,
            "h_flip": h_flip,
            "predictions": {},
            "targets": {}
        }

        for object_idx in range(len(coords_pred)):
            head = coords_pred[object_idx, :2].tolist()
            tip = coords_pred[object_idx, 2:].tolist()

            class_logits = logits_pred[object_idx].item()
            if class_logits > 0.:
                labels["landmarks2d"][f"view_{view_idx}"]["predictions"][f"object_{object_idx}"] = {
                    "head": head,
                    "tip": tip,
                    "screw_prob": class_logits
                }

        for object_idx in range(len(coords_tgt)):
            head = coords_tgt[object_idx, :2].tolist()
            tip = coords_tgt[object_idx, 2:].tolist()

            labels["landmarks2d"][f"view_{view_idx}"]["targets"][f"object_{object_idx}"] = {
                "head": head,
                "tip": tip
            }
