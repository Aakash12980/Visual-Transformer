from comet_ml import Experiment, OfflineExperiment, ExistingOfflineExperiment
from PIL import Image
import ast
import os
import matplotlib.pyplot as plt
import sklearn
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
import pytorch_lightning as pl
from transformers import ViTForImageClassification, AdamW
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import auc, plot_precision_recall_curve
from torchmetrics import PrecisionRecallCurve, AUROC
from torchmetrics.functional import auc
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


comet_params = dict(
    save_dir='./comet',
    offline=True,
    workspace="",
    project_name="",
    api_key="",
    auto_metric_logging=True,
    auto_param_logging=True,
    experiment_name = "transformer_on_brand_code_only_old_data"
)

# experiment = OfflineExperiment(
#     offline_directory='./comet',
#     **comet_params
# )

train_batch_size = 64
eval_batch_size = 64
comet_logger = pl_loggers.CometLogger(**comet_params)

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


class TransformerClassifier(pl.LightningModule):
    def __init__(self, num_labels=1):
        super(TransformerClassifier, self).__init__()
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k',
                                                              num_labels=1,
                                                              id2label={0:"Real"},
                                                              label2id={"Real":0})
        self.sig = nn.Sigmoid()
        # self.preds = []
        # self.labels = []

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        return self.sig(outputs.logits)
        
    def common_step(self, batch, batch_idx, test=False):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.BCELoss()
        # print(logits)
        loss = criterion(torch.reshape(logits, (-1, 1)), torch.reshape(labels, (-1,1)).float())
        predictions = torch.round(torch.squeeze(logits))
        correct = (predictions == labels).sum().item()
        accuracy = correct/pixel_values.shape[0]

        if test:
            # auroc = AUROC(pos_label=1)
            # auc_precision_recall = auroc(torch.squeeze(logits), labels)

            # self.preds.append(torch.squeeze(logits))
            # self.labels.append(labels)
            return (loss, accuracy, logits, labels)
        return loss, accuracy
      
    def training_step(self, batch, batch_idx):
        # if len(self.labels) > 0:
        #     self.preds = []
        #     self.labels = []
        loss, accuracy = self.common_step(batch, batch_idx, test=False)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)
        # self.log("pr_auc", accuracy, on_epoch=True)    
        self.logger.agg_and_log_metrics({"training_loss": loss})   
        self.logger.agg_and_log_metrics({"training_accuracy": accuracy})    

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, logits, labels = self.common_step(batch, batch_idx, True)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)
        # self.log("validation_roc_auc_gathering", roc_auc, on_epoch=True)
        self.logger.agg_and_log_metrics({"validation_loss": loss})   
        self.logger.agg_and_log_metrics({"validation_accuracy": accuracy})    

        return logits, labels

    def validation_epoch_end(self, validation_step_outputs):
        # outputs = torch.stack(validation_step_outputs)
        all_labels = []
        all_preds = []
        all_logits = []
        for out in validation_step_outputs:
            for logits, label in zip(*out):
                predictions = torch.round(torch.squeeze(logits))
                all_labels.append(label.item())
                all_preds.append(predictions.item())
                all_logits.append(torch.squeeze(logits).item())

        pr_curve = PrecisionRecallCurve(pos_label=1)
        # precision, recall, thresholds = pr_curve(torch.cat(self.preds), torch.cat(self.labels))
        # auc_precision_recall = auc(precision, recall, reorder=True)
        precision, recall, thresholds = pr_curve(torch.tensor(all_logits), torch.tensor(all_labels))
        auc_precision_recall = auc(precision, recall, reorder=True)
        self.log("val_pr_auc", auc_precision_recall.item())
        print(f"Val_preds: {len(all_preds)}")


        self.logger.log_metrics({"val_pr_auc": auc_precision_recall.item()})

        return auc_precision_recall


    def test_step(self, batch, batch_idx):
        loss, accuracy, preds, labels = self.common_step(batch, batch_idx, test=True)     
        self.log("Test_loss", loss, on_epoch=True)
        self.log("Test_Accuracy", accuracy, on_epoch=True) 
        # self.log("Test_roc_auc_gathering", roc_auc, on_epoch=True)
        self.logger.agg_and_log_metrics({"test_loss": loss})   
        self.logger.agg_and_log_metrics({"test_accuracy": accuracy}) 

        return preds, labels

    def test_epoch_end(self, outputs):
        all_labels = []
        all_preds = []
        all_logits = []
        category_names = ["Real", "Fake"]
        for out in outputs:
            for logits, label in zip(*out):
                predictions = torch.round(torch.squeeze(logits))
                all_labels.append(label.item())
                all_preds.append(predictions.item())
                all_logits.append(torch.squeeze(logits).item())

        conf_mat = confusion_matrix(all_labels, all_preds)
        print('Confusion Matrix:',conf_mat)

        tpr = fpr = None 
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_labels, all_logits)
        stats = TransformerClassifier.create_auth_report(pd.Series(all_labels), pd.Series(all_logits), reversed(thresholds))

        pd.options.display.float_format = '{:.2f}'.format
        print(stats)
        script_name = '/home/ubuntu/rebag/code/transformer/script'
        stats.to_csv(os.path.join(script_name, 'stats.csv'), index=False)
        TransformerClassifier.draw_sensitivity_and_specificity(stats, self.logger, script_name)
        TransformerClassifier.plot_confusion_matrix(conf_mat, category_names, script_name)

        sensitivity = float(conf_mat[1][1]) / (conf_mat[1][1] + conf_mat[1][0])
        specificity = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])
        trustworthy_real_predictions = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0])
        # self.logger.log_confusion_matrix(matrix=conf_mat, labels=all_labels)

        # self.logger.log_table("stats.csv", tabular_data=stats)
        self.logger.log_metrics({'test_accuracy': sklearn.metrics.accuracy_score(all_labels, all_preds)})
        self.logger.log_metrics({'test_recall': sklearn.metrics.recall_score(all_labels, all_preds)})
        self.logger.log_metrics({'sensitivity_at_0.5': sensitivity})
        self.logger.log_metrics({'specificity_at_0.5': specificity})
        self.logger.log_metrics({'trustworthy_real_predictions': trustworthy_real_predictions})
        if tpr is not None and fpr is not None:
            # self.logger.log_curve("roc-curve", fpr, tpr)
            self.logger.log_metrics({"test-roc-auc": sklearn.metrics.auc(fpr, tpr)})
        # self.logger.log_dataset_info("transformer_models")


        # print(f"Test_preds: {len(all_preds)}")
        # auroc = AUROC(pos_label=1)
        # roc_auc = auroc(torch.tensor(all_logits), torch.tensor(all_labels))
        # self.logger.log_metrics({"test_roc_auc": roc_auc})
        return

    @staticmethod
    def draw_sensitivity_and_specificity(stats, experiment, script_name):
        sensitivity_specificity_path = os.path.join(script_name, "sensitivity_and_specificity.png")
        stats = stats[stats['threshold']<1]
        plt.plot(stats['threshold'], stats['sensitivity'])
        plt.plot(stats['threshold'], stats['specificity'])
        plt.xlabel("Threshold")
        plt.legend(['sensitivity', 'specificity'])
        plt.savefig(sensitivity_specificity_path)
        plt.close()
        # if experiment:
        #     experiment.log_graph(sensitivity_specificity_path, "sensitivity_and_specificity.png")

    @staticmethod
    def plot_confusion_matrix(cnf_mat, category_names, path):
        plt.figure(figsize=(6, 6))
        plt.title("Confusion Matrix")
        sns.heatmap(cnf_mat, annot=True, cbar=False, linewidth=0.5, fmt="d")
        plt.xticks(np.arange(len(category_names)) + 0.5, category_names)
        plt.yticks(np.arange(len(category_names)) + 0.5, category_names)
        plt.ylabel("Actual Category")
        plt.xlabel("Predicted Category")
        plt.savefig(os.path.join(path, 'conf_matrix.png'))

    @staticmethod
    def create_auth_report(y_true, y_pred, threshold):
        y_true = y_true.reset_index(drop=True)
        y_pred = y_pred.reset_index(drop=True)
        filtered_idx = y_pred[y_pred >= 0].index
        y_pred = y_pred.loc[filtered_idx]
        y_true = y_true.loc[filtered_idx]

        def create_cols(row):
            threshold = row['threshold']

            if isinstance(y_pred.iloc[0], np.ndarray):
                y_pred_adjusted = y_pred.apply(lambda prd: np.round(np.mean([0 if x < threshold else 1 for x in prd])))
            else:
                y_pred_adjusted = y_pred.apply(lambda prd: 0 if prd < threshold else 1)

            tp = ((y_true == 1) & (y_pred_adjusted == 1)).sum()
            fn = ((y_true == 1) & (y_pred_adjusted == 0)).sum()
            fp = ((y_true == 0) & (y_pred_adjusted == 1)).sum()
            tn = ((y_true == 0) & (y_pred_adjusted == 0)).sum()

            row['threshold'] = threshold
            row['sensitivity'] = tp / (tp + fn)
            row['specificity'] = tn / (tn + fp)
            row['npv'] = tn / (tn + fn) if tn + fn else None
            row['time_saved'] = (tn + fn) / (tp + fn + fp + tn)
            row['tp'] = int(tp)
            row['fn'] = int(fn)
            row['fp'] = int(fp)
            row['tn'] = int(tn)
            return row

        if isinstance(threshold, float):
            threshold = [threshold]
        data = pd.DataFrame({'threshold': threshold}).apply(create_cols, axis=1)
        data['tp'] = data['tp'].astype(int)
        data['tn'] = data['tn'].astype(int)
        data['fp'] = data['fp'].astype(int)
        data['fn'] = data['fn'].astype(int)
        return data



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
        monitor='validation_loss',
        patience=6,
        strict=False,
        verbose=True,
        mode='min'
    )

    # train_params = {
    #     "logger":comet_logger, "gpus":1, "callbacks":[early_stop_callback, checkpoint_callback], "max_epochs":1, "default_root_dir":'./', "log_every_n_steps"10

    # }

    checkpoint_callback = ModelCheckpoint(monitor="val_pr_auc")

    model = TransformerClassifier()
    trainer = Trainer(logger=comet_logger, gpus=1, callbacks=[early_stop_callback, checkpoint_callback], max_epochs=25, default_root_dir='./', log_every_n_steps=10)
    trainer.fit(model)
    trainer.test(ckpt_path='best')

