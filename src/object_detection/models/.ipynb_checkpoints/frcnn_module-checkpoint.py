from typing import Any, List
import torch
from pytorch_lightning import LightningModule
import torch.nn.functional as F
from object_detection.datamodules import frcnn_datamodule
from itertools import chain
import numpy as np
from object_detection.utils import utils_frcnn
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
import os
import json


class frcnn_module(LightningModule):
    def __init__(
        self,
        net : torch.nn.Module,
        lr: float = 0.001,
        momentum=0.9,
        weight_decay: float = 0.0005,
        nms_thresh : float = 0.2,
        iou_thresh : float = 0.5,
        pred_save_path: str = ""
    ):

        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = net
        self.pred_coll = []
        self.gt_coll = []
        self.nms_thresh = nms_thresh
        self.iou_thresh = iou_thresh
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.pred_save_path = pred_save_path
        
 
    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        images, targets, _ = batch
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        loss_dict = self.net(x=(images, targets))
        loss_classifier = loss_dict['loss_classifier']
        loss_box_reg = loss_dict['loss_box_reg']
        loss_objectness = loss_dict['loss_objectness']
        loss_rpn_box_reg = loss_dict['loss_rpn_box_reg']
        loss = loss_box_reg + loss_objectness + loss_rpn_box_reg + loss_classifier
    
        return loss, loss_box_reg, loss_objectness, loss_rpn_box_reg, loss_classifier

    def training_step(self, batch: Any, batch_idx: int):
        self.net = self.net.train()
        loss, loss_box_reg, loss_objectness, loss_rpn_box_reg, loss_classifier = self.step(batch)
        self.training_step_outputs.append({"loss": loss, "loss_box_reg": loss_box_reg, "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg, "loss_classifier": loss_classifier})
        return {"loss": loss, "loss_box_reg": loss_box_reg, "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg, "loss_classifier": loss_classifier}
        
   
    def on_training_epoch_end(self):

        loss = sum(output['loss'] for output in self.training_step_outputs) / len(self.training_step_outputs)
        loss_box = sum(output['loss_box_reg'] for output in self.training_step_outputs) / len(self.training_step_outputs)
        loss_objectness = sum(output['loss_objectness'] for output in self.training_step_outputs) / len(self.training_step_outputs)
        loss_rpn_box_reg = sum(output['loss_rpn_box_reg'] for output in self.training_step_outputs) / len(self.training_step_outputs)
        loss_classifier = sum(output['loss_classifier'] for output in self.training_step_outputs) / len(self.training_step_outputs)

        print(f'At Training epoch {self.current_epoch}')
        print(f"'Total loss = ' {loss} 'Loss box = ' {loss_box} 'Loss objectness = ' {loss_objectness} 'Loss rpn box reg = ' {loss_rpn_box_reg} 'Loss classifier = ' {loss_classifier} ")
    

        # Logging all the lossed to tensorboard
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_box", loss_box, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_objectness", loss_objectness, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_rpn_box_reg", loss_rpn_box_reg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_classifier", loss_classifier, on_step=False, on_epoch=True, prog_bar=True)
        self.training_step_outputs.clear()

        
    def validation_step(self, batch: Any, batch_idx: int):
        with torch.no_grad():
            self.net = self.net.train()
            loss, loss_box_reg, loss_objectness, loss_rpn_box_reg, loss_classifier = self.step(batch)
        self.validation_step_outputs.append({"loss": loss, "loss_box_reg": loss_box_reg, "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg, "loss_classifier": loss_classifier})

        return {"loss": loss, "loss_box_reg": loss_box_reg, "loss_objectness": loss_objectness, "loss_rpn_box_reg": loss_rpn_box_reg,"loss_classifier": loss_classifier}
        

    def on_validation_epoch_end(self):
        
        loss = sum(output['loss'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)
        loss_box = sum(output['loss_box_reg'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)
        loss_objectness = sum(output['loss_objectness'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)
        loss_rpn_box_reg = sum(output['loss_rpn_box_reg'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)
        loss_classifier = sum(output['loss_classifier'] for output in self.validation_step_outputs) / len(self.validation_step_outputs)

        print(f'At validation epoch {self.current_epoch}')
        print(f"'Total loss = ' {loss} 'Loss box = ' {loss_box} 'Loss objectness = ' {loss_objectness} 'Loss rpn box reg = ' {loss_rpn_box_reg} 'Loss classifier = ' {loss_classifier} ")

        # Logging all the lossed to tensorboard
        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_box", loss_box, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_objectness", loss_objectness, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_rpn_box_reg", loss_rpn_box_reg, on_step=False, on_epoch=True, prog_bar=True)
        self.log("validation/loss_classifier", loss_classifier, on_step=False, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.clear()


    def test_step(self, batch: Any, batch_idx: int):
        
        self.net = self.net.eval()
        images, targets, image_names = batch
        preds = self.net(images)

        # Applying nms on model predicted bounding boxes
        nms_preds = []
        preds_dict = {}
        for i in range(len(preds)):
            nms_pred = utils_frcnn.apply_nms(preds[i], iou_thresh=self.nms_thresh)
            pred_bboxes, scores = nms_pred["boxes"],nms_pred["scores"]
            img_output = {}
            for pred_id in range(scores.size(0)):
                img_output[pred_id] = {}
                img_output[pred_id]['box_id'] = pred_id
                bboxes = [str(it.item()) for it in pred_bboxes[pred_id]]
                img_output[pred_id]['bbox'] = bboxes
                img_output[pred_id]['score'] = str(scores[pred_id].item())
            preds_dict[image_names[i].split('/')[-1]] = img_output

        
        # if not os.path.exists(f"{self.pred_save_path}"):
        #     os.makedirs(f"{self.pred_save_path}")
        # with open(f"{self.pred_save_path}/model_predictions.json", 'w') as f: json.dump(preds_dict, f)
        
        if not os.path.exists(f"{self.pred_save_path}"):
                os.makedirs(f"{self.pred_save_path}")
                
        existing_data = {}
        try:
            with open(f"{self.pred_save_path}/model_predictions.json", 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            pass

        existing_data.update(preds_dict)

        with open(f"{self.pred_save_path}/model_predictions.json", 'w') as f: json.dump(existing_data, f)
        print("Predictions saved...")
            
        self.test_step_outputs.append({"targets": targets, "preds": nms_preds,"image_names": image_names})

        return {"targets": targets, "preds": nms_preds, "image_names": image_names}
   
       
    def on_test_epoch_end(self):
        self.test_step_outputs.clear()
        pass
        
        
    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    