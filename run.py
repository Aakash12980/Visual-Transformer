from comet_ml import Experiment, OfflineExperiment, ExistingOfflineExperiment
from PIL import Image
import ast
import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomRotation,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
# from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers

comet_params = dict(
    workspace="fusemachines",
    project_name="rebag",
    api_key="GPmdVRWMxsPYqrLK5uXYeIK2U",
    auto_metric_logging=True,
    auto_param_logging=True,
)

experiment = OfflineExperiment(
    offline_directory='./comet',
    **comet_params
)

train_batch_size = 64
eval_batch_size = 64
comet_logger = pl_loggers.CometLogger(save_dir="./comet/")

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# print(feature_extractor.size)
img_size = (3,224,224)
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
_train_transforms = Compose(
        [
            RandomResizedCrop(feature_extractor.size),
            # RandomHorizontalFlip(),
            RandomRotation(degrees=(-10, 10)),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )
class CustomDataset(Dataset):
    def __init__(self, df):
        self.img_path_list = df["img_path_list"].values
        self.labels = df["label"].values
        self.size = len(self.labels)
                
    def __getitem__(self, index):
        return self.img_path_list[index], self.labels[index]
    
    def __len__(self):
        return self.size


class Collate:
    def __init__(self, preprocessing="val"):
        self.preprocessing = preprocessing
        
    def __call__(self, batch):
        bag_list=[]
        labels_list = []
        for data in batch:
#             for img_path, label in data:
            img_list = []
            for single_path in data[0]:
                if single_path is not None:
                    img = Image.open(single_path)                    
                    if self.preprocessing=="train":
                        img = _train_transforms(img.convert("RGB"))
                    else:
                        img = _val_transforms(img.convert("RGB"))
                    img_list.append(img.numpy())
                else:
                    img_list.append(np.zeros(img_size))


            img_tensor = torch.as_tensor(img_list[0])  
            bag_list.append(img_tensor.numpy())
            labels_list.append(data[1])
#             yield (img_tensor, label)

        bag_tensor = torch.tensor(bag_list)
        labels_tensor = torch.tensor(labels_list)
        
        return {"pixel_values": bag_tensor, "labels": labels_tensor}


class ViTLightningModule(pl.LightningModule):
    def __init__(self, num_labels=2):
        super(ViTLightningModule, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=2,
                                                              id2label={0:"Real", 1:"Fake"},
                                                              label2id={"Real":0, "Fake":1})

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return outputs.logits
        
    def common_step(self, batch, batch_idx, test=False):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)
        predictions = logits.argmax(-1)
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]
        if test:
            return loss, accuracy, predictions
        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy = self.common_step(batch, batch_idx)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss, accuracy, preds = self.common_step(batch, batch_idx, test=True)     
        self.log("Test_loss", loss, on_epoch=True)
        self.log("Test_Accuracy", accuracy, on_epoch=True)  
        return loss, accuracy, preds

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)

    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader

    def test_dataloader(self):
        return test_dataloader


if __name__ == '__main__':
    train_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_sample_fake_unsample_train.csv")
    val_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_fake_sample_val.csv")
    test_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_fake_sample_test.csv")

    train_df['img_path_list'] = train_df['img_path_list'].apply(lambda x: ast.literal_eval(x))
    val_df['img_path_list'] = val_df['img_path_list'].apply(lambda x: ast.literal_eval(x))
    test_df['img_path_list'] = test_df['img_path_list'].apply(lambda x: ast.literal_eval(x))

    train_df = train_df[train_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]
    val_df = val_df[val_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]
    test_df = test_df[test_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]

    train_df['img_path_list'] = train_df['img_path_list'].apply(lambda x: [x[0]])
    val_df['img_path_list'] = val_df['img_path_list'].apply(lambda x: [x[0]])
    test_df['img_path_list'] = test_df['img_path_list'].apply(lambda x: [x[0]])

    # print(len(train_df))
    # print(train_df['label'].value_counts())
    # exit()

    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    
    train_collate_fn = Collate("train")
    val_collate_fn = Collate()
    test_collate_fn = Collate()

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=train_collate_fn, batch_size=train_batch_size)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=test_collate_fn, batch_size=eval_batch_size)

    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=3,
        strict=False,
        verbose=True,
        mode='min'
    )

    model = ViTLightningModule()
    trainer = Trainer(logger=comet_logger, gpus=1, callbacks=[early_stop_callback], max_epochs=2, default_root_dir='./', log_every_n_steps=10)
    trainer.fit(model)
    print(trainer.test(ckpt_path='best'))

