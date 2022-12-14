from cProfile import label
from comet_ml import Experiment, OfflineExperiment, ExistingOfflineExperiment
from PIL import Image
import ast
import os, cv2
import matplotlib.pyplot as plt
import sklearn
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad
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
# from transformers import TrainingArguments, Trainer
from datasets import load_metric
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pytorch_lightning as pl
from transformers import ViTModel, AdamW
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
import PIL
import numbers


comet_params = dict(
    # save_dir='./comet',
    # offline=True,
    workspace="",
    project_name="",
    api_key="",
    auto_metric_logging=True,
    auto_param_logging=True,
    # experiment_name = "transformer_on_brand_code_only_old_data"
)

experiment = OfflineExperiment(
    offline_directory='./comet',
    **comet_params
)

batch_size = 12
# comet_logger = pl_loggers.CometLogger(**comet_params)

feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
# print(feature_extractor.size)
img_size = (3,224,224)
n_aoi=10
normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad),  int(t_pad),int(r_pad), int(b_pad))
    return padding

class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


_train_transforms = Compose(
        [   
            SquarePad(),
            Resize(feature_extractor.size),
            # CenterCrop(feature_extractor.size),
            # RandomRotation(degrees=(-10, 10)),
            ToTensor(),
            normalize,
        ]
    )

_val_transforms = Compose(
        [  
            SquarePad(),
            Resize(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

class CustomDataset(Dataset):
    def __init__(self, df, preprocessing='val'):
        self.img_path_list = df["img_path_list"].values 
        self.labels = df["label"].values
        self.size = len(self.labels)
        self.preprocessing = preprocessing
                
    def __getitem__(self, index):
        label = self.labels[index]
        img_list = []
        for img_path in self.img_path_list[index]:
            if img_path is not None:
                img = Image.open(img_path)
                w, h = img.size
                if h>w:
                    img = img.rotate(-90, PIL.Image.NEAREST, expand = 1)
                
        
                if self.preprocessing=='train':
                    img = _train_transforms(img.convert("RGB"))
                else:
                    img = _val_transforms(img.convert("RGB"))
                img_list.append(img)
            else:
                img_list.append(torch.zeros((img_size)))
        
        img_tensor = torch.stack(img_list)  
        yield img_tensor.numpy(), label
    
    def __len__(self):
        return self.size
    
    @staticmethod
    def intensity(img):
        transform = transforms.ToPILImage()
        img = transform(img)
        brightness_factor=torch.FloatTensor(1).uniform_(0.4, 0.8).item()
        enhancer = PIL.ImageEnhance.Brightness(img)
        img2 = enhancer.enhance(brightness_factor)
        transform = transforms.ToTensor()
        return transform(img2)
    
    @staticmethod
    def rotate(img):
        ANG_THRSEH = 10
        transform = transforms.ToPILImage()
        img = transform(img)
        ang = torch.FloatTensor(1).uniform_(-ANG_THRSEH, ANG_THRSEH).item()
        transform = transforms.ToTensor()
        return transform(img.rotate(ang, Resampling.NEAREST))


def collate_fn(batch):
    img_list = []
    label_list = []
    for data_gen in batch:
        for img, label in data_gen:
            img_list.append(img)
            label_list.append(label)
    bag_tensor = torch.tensor(img_list)
    labels_tensor = torch.tensor(label_list)
    return {"pixel_values": bag_tensor, "labels": labels_tensor}

class TransformerClassifier(pl.LightningModule):
    def __init__(self, num_labels=1):
        super(TransformerClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dense_1 = nn.Linear(768, 8)
        self.dense_last = nn.Linear(80, 1)
        self.sig = nn.Sigmoid()
        print("Initialised")
        
    def forward(self, pixel_values):
        outputs = [self.vit(pixel_values=pixel_values[:,i], return_dict = False) for i in range(10)]
        d_out_1 = [self.dense_1(outputs[i][1]) for i in range(10)]
        concat = torch.cat(d_out_1,1)
        d_out_last = self.dense_last(concat)
        res = self.sig(d_out_last)
        return res
        
    def common_step(self, batch, batch_idx, test=False):
        pixel_values = batch['pixel_values']
        labels = batch['labels']
        logits = self(pixel_values)

        criterion = nn.BCELoss()
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
        return loss, accuracy, logits, labels
      
    def training_step(self, batch, batch_idx):

        loss, accuracy, logits, labels = self.common_step(batch, batch_idx, test=False)     
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        self.log("training_accuracy", accuracy)

        pr_curve = PrecisionRecallCurve(num_classes = 1, pos_label=1)
        precision, recall, thresholds = pr_curve(logits, labels)
        auc_precision_recall = auc(recall, precision, reorder=True)
        self.log("pr_auc", auc_precision_recall.item())
        print(f" pr_auc: {(auc_precision_recall.item()):>0.6f}%,\n")
        return {'loss':loss , 'accuracy':accuracy, 'pr_auc':auc_precision_recall}
    
    def validation_step(self, batch, batch_idx):
        loss, accuracy, logits, labels = self.common_step(batch, batch_idx, True)     
        self.log("validation_loss", loss, on_epoch=True)
        self.log("validation_accuracy", accuracy, on_epoch=True)   

        return logits, labels

    def validation_epoch_end(self, validation_step_outputs):
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
        precision, recall, thresholds = pr_curve(torch.tensor(all_logits), torch.tensor(all_labels))
        auc_precision_recall = auc(recall, precision, reorder=True)
        self.log("val_pr_auc", auc_precision_recall.item())
        print(f"Val_preds: {len(all_preds)}")
        print(f"Val_pr_auc: {auc_precision_recall.item()}")
        experiment.log_metric("val_pr_auc", auc_precision_recall.item())
        return auc_precision_recall


    def test_step(self, batch, batch_idx):
        loss, accuracy, preds, labels = self.common_step(batch, batch_idx, test=True)     
        self.log("Test_loss", loss, on_epoch=True)
        self.log("Test_Accuracy", accuracy, on_epoch=True)  
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
        labels = ["Real", "Fake"]
        experiment.log_confusion_matrix(matrix=conf_mat, labels=labels)

        tpr = fpr = None 
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(all_labels, all_logits)
        stats = TransformerClassifier.create_auth_report(pd.Series(all_labels), pd.Series(all_logits), reversed(thresholds))

        pd.options.display.float_format = '{:.2f}'.format
        print(stats)
        script_name = '/home/ubuntu/rebag/code/transformer/script'
        stats.to_csv(os.path.join(script_name, 'stats.csv'), index=False)
        TransformerClassifier.draw_sensitivity_and_specificity(stats, script_name)
        TransformerClassifier.plot_confusion_matrix(conf_mat, category_names, script_name)

        sensitivity = float(conf_mat[1][1]) / (conf_mat[1][1] + conf_mat[1][0])
        specificity = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[0][1])
        trustworthy_real_predictions = float(conf_mat[0][0]) / (conf_mat[0][0] + conf_mat[1][0])
        experiment.log_confusion_matrix(matrix=conf_mat, labels=labels)

        experiment.log_table("stats.csv", tabular_data=stats)
        experiment.log_metric('test_accuracy', sklearn.metrics.accuracy_score(all_labels, all_preds))
        experiment.log_metric('test_recall', sklearn.metrics.recall_score(all_labels, all_preds))
        experiment.log_metric('sensitivity_at_0.5', sensitivity)
        experiment.log_metric('specificity_at_0.5', specificity)
        experiment.log_metric('trustworthy_real_predictions', trustworthy_real_predictions)
        if tpr is not None and fpr is not None:
            experiment.log_curve("roc-curve", fpr, tpr)
            experiment.log_metric("test-roc-auc", sklearn.metrics.auc(fpr, tpr))
        experiment.log_dataset_info("transformer_models")

        return

    @staticmethod
    def draw_sensitivity_and_specificity(stats, script_name):
        global experiment
        sensitivity_specificity_path = os.path.join(script_name, "sensitivity_and_specificity.png")
        stats = stats[stats['threshold']<1]
        plt.plot(stats['threshold'], stats['sensitivity'])
        plt.plot(stats['threshold'], stats['specificity'])
        plt.xlabel("Threshold")
        plt.legend(['sensitivity', 'specificity'])
        plt.savefig(sensitivity_specificity_path)
        plt.close()
        print(type(experiment))
        # if experiment:
        experiment.log_image(sensitivity_specificity_path, "sensitivity_and_specificity.png")

    @staticmethod
    def plot_confusion_matrix(cnf_mat, category_names, path):
        global experiment
        plt.figure(figsize=(6, 6))
        plt.title("Confusion Matrix")
        sns.heatmap(cnf_mat, annot=True, cbar=False, linewidth=0.5, fmt="d")
        plt.xticks(np.arange(len(category_names)) + 0.5, category_names)
        plt.yticks(np.arange(len(category_names)) + 0.5, category_names)
        plt.ylabel("Actual Category")
        plt.xlabel("Predicted Category")
        cf_matrix_path = os.path.join(path, 'conf_matrix.png')
        plt.savefig(cf_matrix_path)
        plt.close()
        # experiment.log_image(cf_matrix_path, 'conf_matrix.png')

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
    torch.manual_seed(1)
    train_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_sample_fake_unsample_train.csv")
    val_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_fake_sample_val.csv")
    test_df = pd.read_csv("/home/ubuntu/rebag/code/notebooks/real_fake_sample_test.csv")

    train_df['img_path_list'] = train_df['img_path_list'].apply(lambda x: ast.literal_eval(x))
    val_df['img_path_list'] = val_df['img_path_list'].apply(lambda x: ast.literal_eval(x))
    test_df['img_path_list'] = test_df['img_path_list'].apply(lambda x: ast.literal_eval(x))

    # train_df = train_df[train_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]
    # val_df = val_df[val_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]
    # test_df = test_df[test_df['img_path_list'].apply(lambda x: True if x[0] is not None else False )]

    # train_df['img_path_list'] = train_df['img_path_list'].apply(lambda x: [x[0]])
    # val_df['img_path_list'] = val_df['img_path_list'].apply(lambda x: [x[0]])
    # test_df['img_path_list'] = test_df['img_path_list'].apply(lambda x: [x[0]])

    # print(len(train_df))
    # print(train_df['label'].value_counts())
    # exit()

    train_dataset = CustomDataset(train_df, preprocessing='train')
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=batch_size)


    # import os
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=6,
        strict=False,
        verbose=True,
        mode='min'
    )

    checkpoint_callback = ModelCheckpoint(monitor="val_pr_auc", mode='max')

    model = TransformerClassifier()
    trainer = Trainer(gpus=1, callbacks=[checkpoint_callback], max_epochs=25, default_root_dir='./', log_every_n_steps=10)
    trainer.fit(model)
    trainer.test(ckpt_path='best')
    # trainer.test(model, ckpt_path='/home/ubuntu/rebag/code/transformer/lightning_logs/version_1/checkpoints/epoch=0-step=193.ckpt')

