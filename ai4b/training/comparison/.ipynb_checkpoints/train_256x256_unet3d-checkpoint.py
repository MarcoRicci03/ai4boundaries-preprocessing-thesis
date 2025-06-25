#!/usr/bin/env python
# coding: utf-8

import numpy as np
import rasterio, glob, xarray as xr
import os,sys
import time
from datetime import timedelta

import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

sys.path.append("../../../")
from tfcl.models.ptavit3d.unet3d_multitask import UNet3DMultitask
from tfcl.nn.loss.ftnmt_loss import ftnmt_loss
from tfcl.utils.classification_metric import Classification

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class AI4BNormal_S2(object):
    def __init__(self):
        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03, 0.3]).astype(np.float32)
        self._std_s2 = np.array([3.7141626e+02, 3.8981952e+02, 4.7989127e+02, 9.5173022e+02, 0.2]).astype(np.float32)

    def __call__(self, img):
        # Converte l'immagine in float32
        img = img.astype(np.float32)
        
        # Per serie temporali con forma (c, t, h, w), normalizza lungo la dimensione dei canali
        if img.ndim == 4:  # Serie temporale (c, t, h, w)
            for c in range(img.shape[0]):
                img[c] = (img[c] - self._mean_s2[c]) / self._std_s2[c]
        else:  # Immagine singola (c, h, w)
            for c in range(img.shape[0]):
                img[c] = (img[c] - self._mean_s2[c]) / self._std_s2[c]
        
        return img


class TrainingTransformS2(object):
    def __init__(self, prob=1., mode='train', norm=AI4BNormal_S2()):
        self.geom_trans = A.Compose([
            A.RandomCrop(width=128, height=128, p=1.0),
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.ElasticTransform(p=1),
                A.GridDistortion(distort_limit=0.4, p=1.),
                A.Affine(scale=(0.75, 1.25), translate_percent=0.25, rotate=(-180, 180), p=1.0),
            ], p=1.)
        ],
            additional_targets={'imageS1': 'image', 'mask': 'mask'}, p=prob)

        if mode == 'train':
            self.mytransform = self.transform_train
        elif mode == 'valid':
            self.mytransform = self.transform_valid
        else:
            raise ValueError('transform mode can only be train or valid')
        self.norm = norm

    def transform_valid(self, data):
        timgS2, tmask = data
        if self.norm is not None:
            timgS2 = self.norm(timgS2)
        return timgS2, tmask.astype(np.float32)

    def transform_train(self, data):
        timgS2, tmask = data
        if self.norm is not None:
            timgS2 = self.norm(timgS2)
        tmask = tmask.astype(np.float32)
        c2, t, h, w = timgS2.shape
        timgS2 = timgS2.reshape(c2 * t, h, w)
        result = self.geom_trans(image=timgS2.transpose([1, 2, 0]), mask=tmask.transpose([1, 2, 0]))
        timgS2_t = result['image']
        tmask_t = result['mask']
        timgS2_t = timgS2_t.transpose([2, 0, 1])
        tmask_t = tmask_t.transpose([2, 0, 1])
        c2t, h2, w2 = timgS2_t.shape
        timgS2_t = timgS2_t.reshape(c2, t, h2, w2)
        return timgS2_t, tmask_t

    def __call__(self, *data):
        return self.mytransform(data)


class AI4BDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_data=r'../../../../../datasets/AI4B/sentinel2/',transform=TrainingTransformS2(), mode='train', ntrain=0.9):
        self.flnames_s2_img = sorted(glob.glob(os.path.join(path_to_data, r'images/*/*.nc')))
        self.flnames_s2_mask = sorted(glob.glob(os.path.join(path_to_data, r'masks/*/*.tif')))
        assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError(
            "The number of masks and images do not match.")
        tlen = len(self.flnames_s2_img)
        if mode == 'train':
            self.flnames_s2_img = self.flnames_s2_img[:int(ntrain * tlen)]
            self.flnames_s2_mask = self.flnames_s2_mask[:int(ntrain * tlen)]
        elif mode == 'valid':
            self.flnames_s2_img = self.flnames_s2_img[int(ntrain * tlen):]
            self.flnames_s2_mask = self.flnames_s2_mask[int(ntrain * tlen):]
        else:
            raise ValueError(f"Mode {mode} not understood, should be 'train' or 'valid'.")
        self.transform = transform

    def ds2rstr(self, tname):
        # Definisce le variabili (band) da utilizzare dall'immagine Sentinel-2
        variables2use = ['B2', 'B3', 'B4', 'B8']
        # Apre il file .nc con xarray
        ds = xr.open_dataset(tname)
        
        # Concatena le variabili selezionate
        bands = [ds[var].values[None] for var in variables2use]
        
        # Calcola NDVI: (B8 - B4) / (B8 + B4)
        b8 = ds['B8'].values
        b4 = ds['B4'].values
        ndvi = (b8 - b4) / (b8 + b4 + 1e-8)  # +1e-8 per evitare divisione per zero
        bands.append(ndvi[None])
        
        ds_np = np.concatenate(bands, 0)
        return ds_np

    def read_mask(self, tname):
        return rasterio.open(tname).read((1, 2, 3))

    def __getitem__(self, idx):
        tname_img = self.flnames_s2_img[idx]
        tname_mask = self.flnames_s2_mask[idx]
        timg = self.ds2rstr(tname_img)
        tmask = self.read_mask(tname_mask)
        if self.transform is not None:
            timg, tmask = self.transform(timg, tmask)
        return timg, tmask

    def __len__(self):
        return len(self.flnames_s2_img)

# --- Training Configuration ---
DEBUG=False
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
disable_bar = not sys.stdout.isatty()
# --------------------------

def mtsk_loss(preds, labels, criterion, NClasses=1):
    pred_segm = preds[:, :NClasses]
    pred_bound = preds[:, NClasses:2 * NClasses]
    pred_dists = preds[:, 2 * NClasses:3 * NClasses]
    label_segm = labels[:, :NClasses]
    label_bound = labels[:, NClasses:2 * NClasses]
    label_dists = labels[:, 2 * NClasses:3 * NClasses]
    loss_segm = criterion(pred_segm, label_segm)
    loss_bound = criterion(pred_bound, label_bound)
    loss_dists = criterion(pred_dists, label_dists)
    return (loss_segm + loss_bound + loss_dists) / 3.0

def monitor_epoch(model, epoch, datagen_valid, NClasses=1):
    metric_target = Classification(num_classes=NClasses, task='binary').to(device)
    model.eval()
    tot_loss = 0.0
    count = 0
    criterion = ftnmt_loss()
    
    all_preds_flat = []
    all_labels_flat = []

    valid_pbar = tqdm(datagen_valid, desc=f"Validating Epoch {epoch}", position=1, leave=False, disable=disable_bar)
    for idx, data in enumerate(valid_pbar):
        images, labels = data
        images = images.to(device, non_blocking=True)
        labels = labels.cuda(device, non_blocking=True)
        with torch.inference_mode():
            preds_target = model(images)
            loss = mtsk_loss(preds_target, labels, criterion, NClasses)
        pred_segm = preds_target[:, :NClasses]
        label_segm = labels[:, :NClasses]

        # Model output is already sigmoid-activated probability
        preds_binary = (pred_segm > 0.5).reshape(-1).cpu().numpy()
        labels_binary = label_segm.reshape(-1).cpu().numpy().astype(int)
        all_preds_flat.extend(preds_binary)
        all_labels_flat.extend(labels_binary)

        tot_loss += loss.item()
        count += 1
        metric_target(pred_segm, label_segm)
        if DEBUG and idx > 5:
            break
    metric_kwargs_target = metric_target.compute()
    avg_loss = tot_loss / count if count > 0 else 0.0
    kwargs = {'epoch': epoch, 'tot_val_loss': avg_loss}
    for k, v in metric_kwargs_target.items():
        kwargs[k] = v.cpu().numpy()
    return kwargs, all_preds_flat, all_labels_flat

def plot_metrics(history, name):
    epochs = range(len(history['train_loss']))

    plt.figure(figsize=(20, 8))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Metrics
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['acc'], label='Accuracy')
    plt.plot(epochs, history['kappa'], label='Kappa')
    plt.plot(epochs, history['mcc'], label='MCC')
    plt.plot(epochs, history['precision'], label='Precision')
    plt.plot(epochs, history['recall'], label='Recall')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plot_path = f"output2/{name}_performance_plots.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.savefig(plot_path)
    print(f"Performance plots saved to {plot_path}")
    plt.close()

def evaluate_model(all_labels_flat, all_preds_flat, name):
    print("\n--- Classification Report (Pixel-wise) ---")
    print(classification_report(all_labels_flat, all_preds_flat, target_names=['Background', 'Field']))

    print("\n--- Confusion Matrix (Pixel-wise) ---")
    cm = confusion_matrix(all_labels_flat, all_preds_flat)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Predicted Negative', 'Predicted Positive'], 
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Pixel-wise Confusion Matrix')
    
    cm_path = f"output2/{name}_confusion_matrix.png"
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()

def train(args):
    num_epochs = args.epochs
    batch_size = args.batch_size
    name = args.name
    patience = args.patience
    torch.manual_seed(0)

    NClasses = 1
    nf = 96
    verbose = False

    model = UNet3DMultitask(spatial_size=128).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    train_dataset = AI4BDataset(mode='train')
    valid_dataset = AI4BDataset(mode='valid')
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    criterion = ftnmt_loss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=1e-3, eps=1.e-6)
    scaler = GradScaler()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_metrics = {}
    best_epoch = 0
    best_preds_flat = []
    best_labels_flat = []
    total_train_start_time = time.time()

    history = {
        'train_loss': [], 'val_loss': [], 'acc': [], 'kappa': [],
        'mcc': [], 'precision': [], 'recall': []
    }

    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    for epoch in epoch_pbar:
        model.train()
        epoch_start_time = time.time()
        running_loss = 0
        total_batch_time = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", position=1, leave=False, disable=disable_bar)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            if DEBUG and batch_idx > 5:
                break
            batch_start_time = time.time()
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type='cuda', dtype=torch.float16):
                preds = model(inputs)
                loss = mtsk_loss(preds, labels, criterion, NClasses)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_end_time = time.time()
            total_batch_time += (batch_end_time - batch_start_time)
            running_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        epoch_end_time = time.time()
        epoch_duration = timedelta(seconds=epoch_end_time - epoch_start_time)
        avg_img_time_ms = (total_batch_time / len(train_loader.dataset)) * 1000
        
        kwargs, all_preds_flat, all_labels_flat = monitor_epoch(model, epoch, valid_loader, NClasses=1)
        kwargs['tot_train_loss'] = running_loss / len(train_loader)
        
        history['train_loss'].append(kwargs['tot_train_loss'])
        history['val_loss'].append(kwargs['tot_val_loss'])
        history['acc'].append(kwargs['acc'])
        history['kappa'].append(kwargs['kappa'])
        history['mcc'].append(kwargs['mcc'])
        history['precision'].append(kwargs['precision'])
        history['recall'].append(kwargs['recall'])
        
        print(f"Epoch {epoch} | Duration: {str(epoch_duration).split('.')[0]} | Avg Img Time: {avg_img_time_ms:.2f}ms")

        csv_path = f"output2/{name}.csv"
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, 'a') as f:
            if f.tell() == 0:
                f.write("epoch,tot_train_loss,tot_val_loss,acc,kappa,mcc,precision,recall\n")
            f.write(f"{epoch},{kwargs['tot_train_loss']},{kwargs['tot_val_loss']},{kwargs['acc']},{kwargs['kappa']},{kwargs['mcc']},{kwargs['precision']},{kwargs['recall']}\n")

        if kwargs['tot_val_loss'] < best_val_loss:
            best_val_loss = kwargs['tot_val_loss']
            epochs_no_improve = 0
            best_metrics = kwargs
            best_epoch = epoch
            best_preds_flat = all_preds_flat
            best_labels_flat = all_labels_flat
            model_path = f"output2/model/best_model_{name}.pth"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            state_to_save = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_to_save, model_path)
            print(f"  -> New best model saved with validation loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1

        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            epoch_pbar.write(output_str)

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs.")
            break

    total_train_end_time = time.time()
    total_train_duration = timedelta(seconds=total_train_end_time - total_train_start_time)
    
    print("\n--- Training Summary ---")
    print(f"Total training time: {str(total_train_duration).split('.')[0]}")
    if best_epoch is not None:
        print(f"Best model found at epoch: {best_epoch}")
        print("Best validation metrics:")
        for k, v in best_metrics.items():
            if k not in ['epoch', 'tot_train_loss']:
                value = v.mean() if isinstance(v, np.ndarray) else v
                print(f"  - {k.replace('Binary', '').strip()}: {value:.4f}")
    print("------------------------\n")

    # --- Performance Analysis ---
    if not DEBUG:
        print("\n--- Performance Analysis ---")
        plot_metrics(history, name)

        model_path = f"output2/model/best_model_{name}.pth"
        if os.path.exists(model_path):
            print("\n--- Evaluating Best Model on Validation Set ---")
            evaluate_model(best_labels_flat, best_preds_flat, name)
        else:
            print("\nNo best model was saved. Skipping evaluation.")
        print("--------------------------\n")

def main():
    class Args:
        epochs = 75 if not DEBUG else 5
        batch_size = 64
        patience = 10
        name = "train_256x256_unet3d"
    args = Args()
    train(args)

if __name__ == "__main__":
    main()