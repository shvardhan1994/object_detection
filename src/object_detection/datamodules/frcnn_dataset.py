import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import cv2
from glob import glob


class CustomDataset(Dataset):
    def __init__(self, img_path_list, annot_df, trans=None):
        self.norm = transforms.ToTensor()
        self.trans = trans
        self.img_path_list = img_path_list
        self.annot_df = annot_df

    def __getitem__(self, idx):
        # capture the image name and the full image path
        image_name = self.img_path_list[idx]
        # read the image
        image = cv2.imread(image_name)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.norm(img_rgb) 

        # read the annotations
        path = image_name.split('/')[-1]
        tempdf = self.annot_df[self.annot_df.newpath == path]
        boxes = [tempdf.iloc[i,1:5].tolist() for i in range(len(tempdf))]
        gtboxes = []
        for i in range(len(boxes)):
            if boxes[i][2] > boxes[i][0] and boxes[i][3] > boxes[i][1]:
                gtboxes.append(boxes[i])
            elif boxes[i][0] > boxes[i][2] and boxes[i][1] > boxes[i][3]:
                gtboxes.append([boxes[i][2],boxes[i][3],boxes[i][0],boxes[i][1]])
            elif boxes[i][0] > boxes[i][2] and boxes[i][3] > boxes[i][1]:
                gtboxes.append([boxes[i][2],boxes[i][1],boxes[i][0],boxes[i][3]])
            elif boxes[i][1] > boxes[i][3] and boxes[i][2] > boxes[i][0]:
                gtboxes.append([boxes[i][0],boxes[i][3],boxes[i][2],boxes[i][1]])
        labels = [1] * len(gtboxes)

        # bounding box to tensor
        boxes = torch.as_tensor(gtboxes, dtype=torch.float32)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        image = torch.as_tensor(image, dtype=torch.float32)
        

        return image, target, image_name


    def __len__(self):
        return len(self.img_path_list)