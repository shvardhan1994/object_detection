from distutils.command.config import config
from typing import Optional, Tuple
import os
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from glob import glob
import pathlib
from object_detection.datamodules import frcnn_dataset
import numpy as np
import re
import pandas as pd
import albumentations as A
from object_detection.utils import utils_frcnn



class frcnn_datamodule(LightningDataModule):
    """
    Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        train_data_dir: str = "data/train",
        train_labels_file: str = 'data/train_labels.csv',
        test_data_dir: str = "data/test",
        test_labels_file: str = 'data/test_labels.csv',
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        aug_flag: str = 'N',
        
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.train_input = sorted(glob(train_data_dir+"/*",recursive=True))
        self.train_annot_df = pd.read_csv(train_labels_file, delimiter=';')
        self.test_input = sorted(glob(test_data_dir+"/*",recursive=True))
        self.test_annot_df = pd.read_csv(test_labels_file, delimiter=',')
        self.aug_flag = aug_flag 

        self.train_annot_df['newpath'] = [p.split('/')[-1] for p in self.train_annot_df.Path]
        self.test_annot_df['newpath'] = [p.split('/')[-1] for p in self.test_annot_df.Path]
        
        
        self.train_annot_df = utils_frcnn.correct_annotdf(self.train_annot_df,self.train_input)
        self.test_annot_df = utils_frcnn.correct_annotdf(self.test_annot_df,self.test_input)
        
        # Check if all the input images have corresponding annotations
        self.new_train_input = []
        for i in range(len(self.train_input)):
            path = self.train_input[i].split('/')[-1]
            if len(self.train_annot_df[self.train_annot_df['newpath'] == path]) != 0:
                self.new_train_input.append(self.train_input[i])

        self.new_test_input = []
        for i in range(len(self.test_input)):
            path = self.test_input[i].split('/')[-1]
            if len(self.test_annot_df[self.test_annot_df['newpath'] == path]) != 0:
                self.new_test_input.append(self.test_input[i])


       
    @property
    def num_classes(self) -> int:
        return 2


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`."""

        self.trans = A.Compose([
                            A.HorizontalFlip(0.5),
                            A.VerticalFlip(0.5),
                            A.Rotate(5),
                            # A.Affine(scale = (1.1,1.1),translate_percent = (0.1,0.1), shear=(3,3))
                        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

        self.data_train = frcnn_dataset.CustomDataset(self.new_train_input,self.train_annot_df)
        self.data_test = frcnn_dataset.CustomDataset(self.new_test_input,self.test_annot_df)
        self.data_val = frcnn_dataset.CustomDataset(self.new_test_input,self.test_annot_df)

        print('Number of training images and test images')    
        print(len(self.data_train),len(self.data_test))
        print('Dimension of the image',self.data_train[0][0].shape )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=lambda x:list(zip(*x)))


    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x:list(zip(*x)))

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=lambda x:list(zip(*x)))

