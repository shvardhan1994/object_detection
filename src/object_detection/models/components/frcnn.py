
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

class frcnn(nn.Module):
    def __init__(self, num_classes,rpn_score_thresh=0, box_score_thresh=0):
        """
      
        A FRCNN module performs below operations:
        - Loads the pretrained FasterRCNN model.
        
      """

        super(frcnn, self).__init__()

        self.num_classes = num_classes
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        self.in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(self.in_features, self.num_classes) 

    def forward(self, x, return_all=False):
        if self.model.training: return self.model(x[0], x[1])    
        else:return self.model(x)    
