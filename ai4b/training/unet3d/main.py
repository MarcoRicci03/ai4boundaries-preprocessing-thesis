import numpy as np
import rasterio, glob, xarray as xr
import os, sys
import argparse

import albumentations as A
from albumentations.core.transforms_interface import  ImageOnlyTransform

import torch
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", message=".*cudnnException.*")

# Import the configuration system
from config import Config, build_dataset_path


class AI4BNormal_S2(object):
    """
    Normalizes Sentinel-2 images (CHW format) using predefined mean and standard deviation values.
    
    This class applies channel-wise normalization to Sentinel-2 satellite imagery,
    including the NDVI channel. It supports both single images and time series.
    """

    def __init__(self):
        # Predefined mean values for each Sentinel-2 image channel (including NDVI)
        self._mean_s2 = np.array([5.4418573e+02, 7.6761194e+02, 7.1712860e+02, 2.8561428e+03, 0.3]).astype(np.float32)
        # Predefined standard deviation values for each Sentinel-2 image channel (including NDVI)
        self._std_s2 = np.array([3.7141626e+02, 3.8981952e+02, 4.7989127e+02, 9.5173022e+02, 0.2]).astype(np.float32)

    def __call__(self, img):
        # Convert image to float32
        img = img.astype(np.float32)
        
        # For time series with shape (c, t, h, w), normalize along channel dimension
        if img.ndim == 4:  # Time series (c, t, h, w)
            for c in range(img.shape[0]):
                img[c] = (img[c] - self._mean_s2[c]) / self._std_s2[c]
        else:  # Single image (c, h, w)
            for c in range(img.shape[0]):
                img[c] = (img[c] - self._mean_s2[c]) / self._std_s2[c]
        
        return img


class TrainingTransformS2(object):
    """
    Applies data augmentation transformations to Sentinel-2 images.
    
    This class handles geometric transformations, cropping, and normalization
    for both training and validation phases. It supports time series data
    and multi-target transformations (images and masks).
    """
    def __init__(self, prob=1., mode='train', norm=AI4BNormal_S2(), config=None):
        if config is None:
            config = Config()  # Use default configuration
        
        self.config = config
        
        # Define geometric transformations using Albumentations
        self.geom_trans = A.Compose([
            # Always apply random crop to configurable size
            A.RandomCrop(width=config.image_size, height=config.image_size, p=1.0),
            # Select one of the specified transformations (flip, elastic, distortion, shift-scale-rotate)
            A.OneOf([
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.ElasticTransform(p=1),
                # VERY GOOD - gives perspective projection, really nice and useful - VERY SLOW
                A.GridDistortion(distort_limit=0.4, p=1.),
                A.ShiftScaleRotate(shift_limit=0.25, scale_limit=(0.75, 1.25), rotate_limit=180, p=1.0),
                # Most important Augmentation
            ], p=1.)
        ],
            # Specify additional transformations for other targets: additional image ("imageS1") and mask ("mask")
            additional_targets={'imageS1': 'image', 'mask': 'mask'}, p=prob)

        # Set transformation method based on mode ('train' or 'valid')
        if mode == 'train':
            self.mytransform = self.transform_train
        elif mode == 'valid':
            self.mytransform = self.transform_valid
        else:
            raise ValueError('transform mode can only be train or valid')
        self.norm = norm

    def transform_valid(self, data):
        timgS2, tmask = data  # For validation: data is a tuple containing Sentinel-2 image and mask

        # Apply normalization if available
        if self.norm is not None:
            timgS2 = self.norm(timgS2)

         # Convert mask to float32 and return data
        return timgS2, tmask.astype(np.float32)

    def transform_train(self, data):
        timgS2, tmask = data # For training: data is a tuple containing Sentinel-2 image and mask

        # Apply normalization to the image if available
        if self.norm is not None:
            timgS2 = self.norm(timgS2)

        # Convert mask to float32
        tmask = tmask.astype(np.float32)

        # Special handling for time series:
        # The image has shape (c2, t, h, w) where:
        # c2 = number of channels, t = number of time steps, h = height, w = width
        c2, t, h, w = timgS2.shape

        # Reshape the image to (c2*t, h, w) to facilitate transformation application
        timgS2 = timgS2.reshape(c2 * t, h, w)

        # Albumentations requires that the image has channels in the last position:
        # Transpose image and mask from (c2*t, h, w) to (h, w, c2*t)
        result = self.geom_trans(image=timgS2.transpose([1, 2, 0]), mask=tmask.transpose([1, 2, 0]))

        # Retrieve the transformed image and mask
        timgS2_t = result['image']
        tmask_t = result['mask']

        # Bring channels back to first dimension:
        # Transpose image and mask from (h, w, c2*t) to (c2*t, h, w)
        timgS2_t = timgS2_t.transpose([2, 0, 1])
        tmask_t = tmask_t.transpose([2, 0, 1])

        # Get new dimensions after transformation
        c2t, h2, w2 = timgS2_t.shape

        # Reshape the transformed image to restore original shape (c2, t, h2, w2)
        timgS2_t = timgS2_t.reshape(c2, t, h2, w2)

        return timgS2_t, tmask_t # Return transformed image and mask

    def __call__(self, *data):
        return self.mytransform(data)  # Allows calling the class instance as a function, applying the appropriate transformation



class AI4BDataset(torch.utils.data.Dataset):
    """
    Custom dataset for Sentinel-2 images (AI4Boundaries).
    
    This dataset handles loading, preprocessing, and augmentation of Sentinel-2 satellite
    images and their corresponding masks for boundary detection tasks.
    """    
    def __init__(self, path_to_data, transform=None, mode='train', ntrain=0.9, config=None):
        if config is None:
            config = Config()  # Use default configuration
            
        self.config = config
        
        # Build the correct dataset path based on resolution
        if hasattr(config, 'dataset_resolution'):
            actual_path = build_dataset_path(path_to_data, config.dataset_resolution)
        else:
            actual_path = path_to_data
        
        if transform is None:
            transform = TrainingTransformS2(mode=mode, config=config)

        # Search and sort in ascending order the Sentinel-2 image files (.nc)
        self.flnames_s2_img = sorted(glob.glob(os.path.join(actual_path, r'images/*/*.nc')))

        # Search and sort in ascending order the mask files (.tif)
        self.flnames_s2_mask = sorted(glob.glob(os.path.join(actual_path, r'masks/*/*.tif')))

        # Verify that the number of images and masks is equal
        assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError(
            "Some problem, the masks and images are not in the same numbers, aborting")

        print(f"🗂️  Dataset path: {actual_path}")
        print(f"📊 Dataset resolution: {getattr(config, 'dataset_resolution', 'default')}")
        if hasattr(config, 'crop_size'):
            print(f"✂️  Crop size: {config.crop_size}x{config.crop_size}")

        tlen = len(self.flnames_s2_img)

        # Split the data based on mode: training or validation
        if mode == 'train': # For training, use the first ntrain% of the data
            self.flnames_s2_img = self.flnames_s2_img[:int(ntrain * tlen)]
            self.flnames_s2_mask = self.flnames_s2_mask[:int(ntrain * tlen)]
        elif mode == 'valid': # For validation, use the remaining 1 - ntrain% of the data
            self.flnames_s2_img = self.flnames_s2_img[int(ntrain * tlen):]
            self.flnames_s2_mask = self.flnames_s2_mask[int(ntrain * tlen):]
        else:
            raise ValueError("Cannot undertand mode::{}, should be either train or valid, aborting...".format(mode))

        # Save the transformation (data augmentation + normalization) to apply to the data
        self.transform = transform

        # Print the dataset size and the number of images and masks
        print(f"Dataset {mode} size: {len(self.flnames_s2_img)} images, {len(self.flnames_s2_mask)} masks")
        # Verify that the images and masks are in equal number
        assert len(self.flnames_s2_img) == len(self.flnames_s2_mask), ValueError(
            "Some problem, the masks and images are not in the same numbers, aborting")
    
    def ds2rstr(self, tname):
        """
        Helper function to read NetCDF files and convert them to raster format.
        
        This function handles NDVI calculation based on configuration:
        - If use_precomputed_ndvi is True and NDVI exists in the dataset, use it directly
        - Otherwise, calculate NDVI on-the-fly from B4 and B8 bands
        
        Args:
            tname (str): Path to the NetCDF file
            
        Returns:
            numpy.ndarray: Array with shape (channels, time, height, width)
        """
        # Use configured bands
        variables2use = getattr(self.config, 's2_bands', ['B2', 'B3', 'B4', 'B8'])
        ndvi_epsilon = getattr(self.config, 'ndvi_epsilon', 1e-8)
        use_precomputed_ndvi = getattr(self.config, 'use_precomputed_ndvi', False)
        
        # Open the .nc file with xarray
        ds = xr.open_dataset(tname)
        
        if use_precomputed_ndvi and 'NDVI' in ds.variables:
            # Use pre-computed NDVI if available
            bands = [ds[var].values[None] for var in variables2use]
        else:
            # Calculate NDVI on-the-fly
            base_bands = ['B2', 'B3', 'B4', 'B8']
            bands = [ds[var].values[None] for var in base_bands]
            
            # Calculate NDVI: (B8 - B4) / (B8 + B4)
            b8 = ds['B8'].values
            b4 = ds['B4'].values
            ndvi = (b8 - b4) / (b8 + b4 + ndvi_epsilon)  # Use configurable epsilon
            bands.append(ndvi[None])
        
        ds_np = np.concatenate(bands, 0)
        return ds_np

    def read_mask(self, tname):
        """
        Read mask file using rasterio.
        
        Args:
            tname (str): Path to the mask file
            
        Returns:
            numpy.ndarray: Mask array with shape (3, height, width)
        """
        # Open the mask file with rasterio and read channels 1, 2 and 3
        return rasterio.open(tname).read((1, 2, 3))

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx (int): Index of the item to retrieve
            
        Returns:
            tuple: (image, mask) where both are processed according to the configuration
        """
        # Get the path of the image and mask files corresponding to index idx
        tname_img = self.flnames_s2_img[idx]
        tname_mask = self.flnames_s2_mask[idx]

        # Load the image and mask from .nc files
        timg = self.ds2rstr(tname_img)
        tmask = self.read_mask(tname_mask)

         # Apply transformation (data augmentation, normalization, etc.) if defined
        if self.transform is not None:
            timg, tmask = self.transform(timg, tmask)

        return timg, tmask # Return processed image and mask

    def __len__(self):
        """
        Get the total number of images in the dataset.
        
        Returns:
            int: Number of images in the dataset
        """
        # Return the total number of images present in the dataset
        return len(self.flnames_s2_img)


# Add a path to the system path list for any additional module imports
sys.path.append("../../../")

from torch.utils.data import DataLoader

from tfcl.models.ptavit3d.unet3d_multitask import UNet3DMultitask
from tfcl.nn.loss.ftnmt_loss import ftnmt_loss
from tfcl.utils.classification_metric import Classification
from datetime import datetime


def mtsk_loss(preds, labels, criterion, NClasses=1):
    """
    Function that calculates the multitask loss for three tasks:
    - segmentation
    - boundary detection  
    - distance prediction
    
    Args:
        preds (torch.Tensor): Model predictions
        labels (torch.Tensor): Ground truth labels
        criterion: Loss function to use
        NClasses (int): Number of classes (default: 1)
        
    Returns:
        torch.Tensor: Average loss across the three tasks
    """

    # Extract predictions for the segmentation task
    pred_segm = preds[:, :NClasses]
    # Extract predictions for the boundary detection task
    pred_bound = preds[:, NClasses:2 * NClasses]
    # Extract predictions for the distance prediction task
    pred_dists = preds[:, 2 * NClasses:3 * NClasses]

    # Extract labels for the segmentation task
    label_segm = labels[:, :NClasses]
    # Extract labels for the boundary detection task
    label_bound = labels[:, NClasses:2 * NClasses]
    # Extract labels for the distance prediction task
    label_dists = labels[:, 2 * NClasses:3 * NClasses]

    # Calculate the loss for each task using the provided criterion (loss function)
    loss_segm = criterion(pred_segm, label_segm)
    loss_bound = criterion(pred_bound, label_bound)
    loss_dists = criterion(pred_dists, label_dists)

    # Return the average loss of the three tasks
    return (loss_segm + loss_bound + loss_dists) / 3.0

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

def monitor_epoch(model, epoch, datagen_valid, config, NClasses=1, device='cuda', should_disable_bar=False):
    """
    Monitor and evaluate the model performance during validation.
    
    This function runs the model on validation data and computes various metrics
    including loss, accuracy, kappa, MCC, precision, and recall.
    
    Args:
        model: The neural network model to evaluate
        epoch (int): Current epoch number
        datagen_valid: Validation data loader
        config: Configuration object with debug settings
        NClasses (int): Number of classes (default: 1)
        device (str): Device to run on ('cuda' or 'cpu')
        should_disable_bar (bool): Whether to disable progress bar
        
    Returns:
        dict: Dictionary containing computed metrics
    """
    # Create instance for monitoring classification metrics (binary task)
    metric_target = Classification(num_classes=NClasses, task='binary').to(device)
    model.eval() # Set model to evaluation mode

    tot_loss = 0.0
    count = 0

    criterion = ftnmt_loss()

    # Create progress bar for validation loop
    valid_pbar = tqdm(datagen_valid, desc=f"Validating Epoch {epoch}", position=1, leave=False, disable=should_disable_bar)
    for idx, data in enumerate(valid_pbar):
        images, labels = data                   # Extract images and labels from datagen_valid
        images = images.to(device, non_blocking=True) # Move images to GPU
        labels = labels.cuda(device, non_blocking=True) # Move labels to GPU

        # Run model in inference mode (without gradient computation)
        with torch.inference_mode():
            preds_target = model(images)
            loss = mtsk_loss(preds_target, labels, criterion, NClasses)

        pred_segm = preds_target[:, :NClasses] # Select predictions for segmentation task (first NClasses columns)
        label_segm = labels[:, :NClasses]      # Select labels for segmentation task (first NClasses columns)

        # Calculate loss for current batch and accumulate the value
        tot_loss += loss.item()
        count += 1

        metric_target(pred_segm, label_segm) # Update metric with current predictions and labels

        # If DEBUG is active and more than debug_batches have been processed, break the loop (debug option)
        if config.debug and idx > config.debug_batches:
            break

    # Calculate aggregate metrics after validation loop
    metric_kwargs_target = metric_target.compute()

    # Calculate average loss
    avg_loss = tot_loss / count if count > 0 else 0.0

    kwargs = {'epoch': epoch, 'tot_val_loss': avg_loss}
    for k, v in metric_kwargs_target.items():
        kwargs[k] = v.cpu().numpy()
    return kwargs


def train(args):
    """
    Main training function that orchestrates the entire training process.
    
    This function handles:
    - Configuration setup
    - Model initialization
    - Dataset and dataloader creation
    - Training loop with validation
    - Model saving and early stopping
    
    Args:
        args: Parsed command line arguments
    """
    # Handle compatibility with the old image_size parameter
    if hasattr(args, 'image_size') and args.image_size is not None:
        print("⚠️  WARNING: --image_size is deprecated. Use --crop_size instead.")
        if args.crop_size is None:
            args.crop_size = args.image_size
    # Create configuration with provided parameters
    config = Config(
        dataset_resolution=getattr(args, 'dataset_resolution', 512),
        crop_size=getattr(args, 'crop_size', None),
        debug=getattr(args, 'debug', False),
        debug_batches=getattr(args, 'debug_batches', 5),
        use_precomputed_ndvi=getattr(args, 'use_precomputed_ndvi', False),
        use_mixed_precision=getattr(args, 'use_mixed_precision', False)
    )
    # Extract training parameters from arguments
    num_epochs = args.epochs
    batch_size = args.batch_size
    name = args.name
    patience = args.patience
    data_path = args.data_path
    
    # Create output directory with timestamp and name
    output_dir = f"outputs/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    
    device_ids = args.device_ids
    lr = args.learning_rate
    
    # Print the configuration being used    
    print(f"🚀 Training Configuration:")
    print(f"  📁 Dataset resolution: {config.dataset_resolution}x{config.dataset_resolution}")
    print(f"  ✂️  Crop size: {config.crop_size}x{config.crop_size}")
    print(f"  🐛 Debug mode: {'Active' if config.debug else 'Inactive'}")
    print(f"  🌿 NDVI: {'Pre-computed' if config.use_precomputed_ndvi else 'Calculated on-the-fly'}")
    print(f"  ⚡ Mixed precision: {'Enabled' if config.use_mixed_precision else 'Disabled'}")
    if config.debug:
        print(f"  📦 Debug batches: {config.debug_batches}")
    print(f"  📏 Batch size: {batch_size}")
    print(f"  🎯 GPU devices: {device_ids}")
    print("="*50)

    # Determine whether to disable progress bar
    is_interactive = sys.stdout.isatty()
    should_disable_bar = args.disable_bar or not is_interactive

    torch.manual_seed(0)

    # Model configuration
    NClasses = 1
    nf = 96
    verbose = False

    device = torch.device(f'cuda:{device_ids[0]}' if torch.cuda.is_available() else 'cpu')
    model = UNet3DMultitask(spatial_size=config.image_size).to(device)  # Use configuration dimension

    if torch.cuda.device_count() > 1 and len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # Dataset and Dataloader setup
    train_dataset = AI4BDataset(mode='train', path_to_data=data_path, config=config)
    valid_dataset = AI4BDataset(mode='valid', path_to_data=data_path, config=config)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # DataLoader optimizations
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True,  # Shuffle for training
        num_workers=0, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        dataset=valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0, 
        pin_memory=True
    )

    # Loss function and optimizer
    criterion = ftnmt_loss()
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr, eps=1.e-6)

    scaler = GradScaler()

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, disable=should_disable_bar)
    for epoch in epoch_pbar:
        model.train()
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", position=1, leave=False, disable=should_disable_bar)
        running_loss = 0
        batch_count = 0        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            # Break early if in debug mode and batch limit reached
            if config.debug and batch_idx > config.debug_batches:
                  break
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)       
            optimizer.zero_grad(set_to_none=True)

            # Dynamic mixed precision training based on configuration
            if config.use_mixed_precision:
                # Use mixed precision (FP16) for better performance
                with autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(inputs)
                    loss = mtsk_loss(preds, labels, criterion, NClasses)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard FP32 training
                preds = model(inputs)
                loss = mtsk_loss(preds, labels, criterion, NClasses)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            batch_count += 1
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            # ==========================================================
            # === NUOVO BLOCCO DI CODICE PER L'HEARTBEAT ===
            # ==========================================================
            log_interval = 10  # Stampa un messaggio ogni 10 batch. Puoi cambiare questo valore.
            if (batch_idx + 1) % log_interval == 0 and should_disable_bar:
                total_batches = len(train_loader) # Numero totale di batch nell'epoca
                print(f"  [Epoch {epoch}] -> Batch {batch_idx + 1}/{total_batches} processato. Loss attuale: {loss.item():.4f}")
            # ==========================================================
            # === FINE DEL NUOVO BLOCCO ===
            # ==========================================================
        
        # Validation and metric computation
        kwargs = monitor_epoch(model, epoch, valid_loader, config, NClasses=1, device=device, should_disable_bar=should_disable_bar)
        # Calculate average training loss
        kwargs['tot_train_loss'] = running_loss / batch_count if batch_count > 0 else 0.0
        
        # Write to CSV file: epoch number, loss, etc.
        with open(f"{output_dir}/{name}.csv", 'a') as f:
            # Write header if file is empty
            if os.stat(f.name).st_size == 0:
                f.write("epoch,tot_train_loss,tot_val_loss,acc,kappa,mcc,precision,recall\n")
            
            # Handle NaN and infinite values
            def format_value(val):
                if isinstance(val, (int, float)):
                    if np.isnan(val) or np.isinf(val):
                        return "NaN"
                    return f"{val:.6f}"
                return str(val)
            
            # Write formatted values
            f.write(f"{epoch}," +
                   f"{format_value(kwargs['tot_train_loss'])}," +
                   f"{format_value(kwargs['tot_val_loss'])}," +
                   f"{format_value(kwargs['acc'])}," +
                   f"{format_value(kwargs['kappa'])}," +
                   f"{format_value(kwargs['mcc'])}," +
                   f"{format_value(kwargs['precision'])}," +
                   f"{format_value(kwargs['recall'])}\n")

        # Model saving and early stopping logic
        if kwargs['tot_val_loss'] < best_val_loss:
            best_val_loss = kwargs['tot_val_loss']
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model_{name}.pth")
            print(f"  --> Best model saved (Loss improved to {kwargs['tot_val_loss']:.4f}).")
        else:
            epochs_no_improve += 1

        if verbose:
            output_str = ', '.join(f'{k}:: {v}, |===|, ' for k, v in kwargs.items())
            epoch_pbar.write(output_str)

        if epochs_no_improve >= patience:
            print("Early stopping triggered. Training stopped.")
            break

def parse_args():
    """
    Parse command line arguments for the training script.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='AI4Boundaries Training Script')
    parser.add_argument('--epochs', type=int, default=75, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--name', type=str, required=True, help='Experiment name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device_ids', nargs='+', type=int, default=[0], help='GPU device IDs to use')
    parser.add_argument('--disable_bar', action='store_true', help='Add this flag to disable progress bar.')
    
    # New parameters for better configuration management
    parser.add_argument('--dataset_resolution', type=int, choices=[256, 1024], default=1024, 
                        help='Dataset resolution to use: 256 or 1024 (default: 1024)')
    parser.add_argument('--crop_size', type=int, 
                        help='Crop/patch size (default: same as dataset_resolution)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode (process only a few batches)')
    parser.add_argument('--debug_batches', type=int, default=5, help='Number of batches to process in debug mode (default: 5)')
    parser.add_argument('--use_precomputed_ndvi', action='store_true', 
                        help='Use pre-computed NDVI from dataset instead of calculating on-the-fly')
    parser.add_argument('--use_mixed_precision', action='store_true', 
                        help='Enable mixed precision training (FP16) for better performance and memory usage')
    
    # Keep image_size for compatibility (deprecated)
    parser.add_argument('--image_size', type=int, 
                        help='[DEPRECATED] Use --crop_size instead. Image size')
    
    return parser.parse_args()

def main():
    """
    Main function that handles argument parsing and training execution.
    """
    args = parse_args()
    try:
        train(args)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[OOM] Batch size {args.batch_size} failed due to out-of-memory error. Please lower the batch size and try again.")
        else:
            raise

if __name__ == "__main__":
    import logging

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

    main()
    
    

# === EXAMPLE USAGE COMMANDS ===

# DATASET 1024x1024 (High Resolution)
# python main.py --epochs 75 --batch_size 8 --patience 15 --name test_1024 --data_path /path/to/dataset --learning_rate 1e-3 --device_ids 0 --dataset_resolution 1024 --crop_size 512 --use_precomputed_ndvi

# DATASET 256x256 (Fast Training with Mixed Precision)  
# python main.py --epochs 75 --batch_size 16 --patience 15 --name test_256_fp16 --data_path /path/to/dataset --learning_rate 1e-3 --device_ids 0 --dataset_resolution 256 --crop_size 128 --use_precomputed_ndvi --use_mixed_precision

# DEBUG MODE (Quick Test)
# python main.py --debug --debug_batches 2 --epochs 1 --dataset_resolution 256 --crop_size 64 --use_mixed_precision --name debug_test --data_path /path/to/dataset

# HIGH PERFORMANCE (Mixed Precision + Optimizations)
# python main.py --epochs 100 --batch_size 12 --dataset_resolution 1024 --crop_size 512 --use_mixed_precision --use_precomputed_ndvi --patience 20 --name high_performance --data_path /path/to/dataset
