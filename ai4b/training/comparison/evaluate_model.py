#!/usr/bin/env python
# coding: utf-8

import numpy as np
import rasterio, glob, xarray as xr
import os,sys
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import OrderedDict

import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# Aggiunge il percorso per trovare i moduli tfcl, come nello script originale
sys.path.append("../../../")
from tfcl.models.ptavit3d.ptavit3d_dn import ptavit3d_dn

class AI4BNormal_S2(object):
    def __init__(self):
        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03]).astype(np.float32)
        self._std_s2 = np.array([3.7141626e+02, 3.8981952e+02, 4.7989127e+02, 9.5173022e+02]).astype(np.float32)

    def __call__(self, img):
        img = img.astype(np.float32).T
        img -= self._mean_s2
        img /= self._std_s2
        return img.T


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
    def __init__(self, path_to_data=r'../../../../../datasets/AI4B/sentinel2/',transform=None, mode='train', ntrain=0.9):
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
        
        if transform is None:
            self.transform = TrainingTransformS2(mode=mode)
        else:
            self.transform = transform

    def ds2rstr(self, tname):
        variables2use = ['B2', 'B3', 'B4', 'B8']
        ds = xr.open_dataset(tname)
        ds_np = np.concatenate([ds[var].values[None] for var in variables2use], 0)
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

def evaluate_model(model, dataloader, device, name, NClasses=1):
    model.eval()
    all_preds_flat = []
    all_labels_flat = []

    eval_pbar = tqdm(dataloader, desc="Evaluating best model", leave=False)
    with torch.inference_mode():
        for images, labels in eval_pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            preds_target = model(images)
            pred_segm = preds_target[:, :NClasses]
            label_segm = labels[:, :NClasses]

            # L'output del modello è una probabilità (sigmoid), converto in binario
            # Uso .reshape() come da sua correzione per evitare l'errore di stride
            preds_binary = (pred_segm > 0.5).reshape(-1).cpu().numpy()
            labels_binary = label_segm.reshape(-1).cpu().numpy().astype(int)

            all_preds_flat.extend(preds_binary)
            all_labels_flat.extend(labels_binary)

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
    
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, f"{name}_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix plot saved to {cm_path}")
    plt.close()

def main():
    # --- Configuration ---
    name = "train_256x256_ptavit3d_dn"
    batch_size = 8
    NClasses = 1
    # Imposta il device (GPU o CPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Model Configuration ---
    nf = 96
    model_config = {'in_channels': 4, 'spatial_size_init': (128, 128), 'depths': [2, 2, 5, 2],
                    'nfilters_init': nf, 'nheads_start': nf // 4, 'NClasses': NClasses,
                    'verbose': False, 'segm_act': 'sigmoid'}

    # --- Load Model ---
    model = ptavit3d_dn(**model_config)
    model_path = os.path.join("output", "model", f"best_model_{name}.pth")

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        print("Please ensure the trained model exists at the correct path.")
        return

    print(f"Loading model from {model_path}")
    state_dict = torch.load(model_path, map_location=device)

    # Gestisce il caso in cui il modello sia stato salvato con DataParallel (multi-GPU)
    if list(state_dict.keys())[0].startswith('module.'):
        print("Model was saved with DataParallel. Removing 'module.' prefix.")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name_key = k[7:] # remove `module.`
            new_state_dict[name_key] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
    
    model.to(device)

    # --- Create Validation Dataloader ---
    # Applica le trasformazioni corrette per la validazione (senza augmentation)
    valid_transform = TrainingTransformS2(mode='valid')
    valid_dataset = AI4BDataset(mode='valid', transform=valid_transform)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # --- Run Evaluation ---
    print("\n--- Evaluating Best Model on Validation Set ---")
    evaluate_model(model, valid_loader, device, name, NClasses=NClasses)
    print("\n--- Evaluation Finished ---")


if __name__ == "__main__":
    main()