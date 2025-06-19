# Configuration file for AI4Boundaries Training Script

class Config:
    """
    Configuration class for global parameters of the AI4Boundaries project.
    
    This class manages all the configuration parameters needed for training,
    including dataset resolution, crop sizes, debug settings, and NDVI handling.
    
    Attributes:
        dataset_resolution (int): Dataset resolution to use (256 or 1024)
        crop_size (int): Size of the crop/patch from dataset images
        debug (bool): Debug mode that limits the number of processed batches (default: False)
        debug_batches (int): Number of batches to process in debug mode (default: 5)
        use_precomputed_ndvi (bool): If True, use pre-computed NDVI instead of calculating it (default: False)
    """
    def __init__(self, dataset_resolution=512, crop_size=None, debug=False, debug_batches=5, use_precomputed_ndvi=False, use_mixed_precision=False):
        """
        Initialize the configuration with the provided parameters.
        
        Args:
            dataset_resolution (int): Dataset resolution to use (256 or 1024)
            crop_size (int, optional): Crop/patch size. If None, uses dataset_resolution
            debug (bool): Enable/disable debug mode
            debug_batches (int): Number of batches in debug mode
            use_precomputed_ndvi (bool): If True, use pre-computed NDVI from dataset
            use_mixed_precision (bool): If True, use mixed precision training (FP16) for better performance
        """        
        # Set basic configuration parameters
        self.dataset_resolution = dataset_resolution
        self.crop_size = crop_size if crop_size is not None else dataset_resolution
        self.debug = debug
        self.debug_batches = debug_batches
        self.use_precomputed_ndvi = use_precomputed_ndvi
        self.use_mixed_precision = use_mixed_precision
        
        # Configure Sentinel-2 bands based on NDVI usage
        if use_precomputed_ndvi:
            # If using pre-computed NDVI, include the NDVI channel
            self.s2_bands = ['B2', 'B3', 'B4', 'B8', 'NDVI']
        else:
            # Otherwise use standard bands to calculate NDVI on-the-fly
            self.s2_bands = ['B2', 'B3', 'B4', 'B8']
        
        # Small epsilon value to avoid division by zero in NDVI calculation
        self.ndvi_epsilon = 1e-8
        
        # Validate configuration parameters
        if self.dataset_resolution not in [256, 1024]:
            raise ValueError("dataset_resolution must be 256 or 1024")
        if self.crop_size <= 0:
            raise ValueError("crop_size must be greater than 0")
        if self.crop_size > self.dataset_resolution:
            raise ValueError(f"crop_size ({self.crop_size}) cannot be greater than dataset_resolution ({self.dataset_resolution})")        
        if self.debug_batches <= 0:
            raise ValueError("debug_batches must be greater than 0")
    
    @property
    def image_size(self):
        """Alias for crop_size for compatibility with existing code."""
        return self.crop_size
    
    @property
    def input_channels(self):
        """Returns the number of input channels (always 5: B2,B3,B4,B8,NDVI)."""
        return 5  # Always 5 total channels
    
    @property
    def expected_dataset_channels(self):
        """Returns the number of channels expected from the dataset."""
        return 5 if self.use_precomputed_ndvi else 4
    
    def get_dataset_path_suffix(self):
        """Returns the path suffix for the correct dataset."""
        return f"_{self.dataset_resolution}" if self.dataset_resolution != 512 else ""
    
    def set_debug(self, debug_mode, debug_batches=None):
        """
        Set debug mode configuration.
        
        Args:
            debug_mode (bool): Enable/disable debug mode
            debug_batches (int, optional): Number of batches for debug mode
        """
        self.debug = debug_mode
        if debug_batches is not None:
            self.debug_batches = debug_batches
    
    def get_info(self):
        """
        Returns a formatted string with configuration information.
        
        Returns:
            str: Formatted configuration information
        """
        info = f"AI4Boundaries Configuration:\n"
        info += f"  - Dataset resolution: {self.dataset_resolution}x{self.dataset_resolution}\n"
        info += f"  - Crop size: {self.crop_size}x{self.crop_size}\n"
        info += f"  - Debug mode: {'Active' if self.debug else 'Inactive'}\n"
        info += f"  - Pre-computed NDVI: {'Yes' if self.use_precomputed_ndvi else 'No (calculated on-the-fly)'}\n"
        info += f"  - Input channels: {self.input_channels} ({', '.join(self.s2_bands)})\n"
        if self.debug:
            info += f"  - Debug batches: {self.debug_batches}\n"
        return info
    
    def __str__(self):
        """String representation of the configuration."""
        return self.get_info()


class DefaultConfig:
    """
    Default configuration for AI4Boundaries training.
    This class can be extended or modified for different configurations.
    """
    
    # === IMAGE CONFIGURATIONS ===
    IMAGE_SIZE = 512
    
    # === DEBUG CONFIGURATIONS ===
    DEBUG = False
    DEBUG_MAX_BATCHES = 5
    
    # === MODEL CONFIGURATIONS ===
    NUM_CLASSES = 1
    MODEL_NF = 96  # Number of features in the model
    
    # === TRAINING CONFIGURATIONS ===
    TRAIN_SPLIT = 0.9  # 90% for training, 10% for validation
    RANDOM_SEED = 0
    
    # === DATALOADER CONFIGURATIONS ===
    NUM_WORKERS = 0  # For Windows, often better to use 0
    PIN_MEMORY = True
    
    # === OPTIMIZATION CONFIGURATIONS ===
    OPTIMIZER_EPS = 1e-6
    USE_MIXED_PRECISION = False
    
    # === SENTINEL-2 CONFIGURATIONS ===
    S2_BANDS = ['B2', 'B3', 'B4', 'B8']
    NDVI_EPSILON = 1e-8  # To avoid division by zero
    
    # === AUGMENTATION CONFIGURATIONS ===
    AUGMENTATION_PROB = 1.0
    GRID_DISTORTION_LIMIT = 0.4
    SHIFT_LIMIT = 0.25
    SCALE_LIMIT = (0.75, 1.25)
    ROTATE_LIMIT = 180
    
    # === LOGGING CONFIGURATIONS ===
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'training.log'


class Dataset256Config(DefaultConfig):
    """
    Configuration for 256x256 dataset with various crop sizes.
    """
    DATASET_RESOLUTION = 256
    IMAGE_SIZE = 128  # Default crop size equal to dataset
    

class Dataset1024Config(DefaultConfig):
    """
    Configuration for 1024x1024 dataset with various crop sizes.
    """
    DATASET_RESOLUTION = 1024
    IMAGE_SIZE = 512  # Default crop size smaller than dataset


class Dataset256DebugConfig(Dataset256Config):
    """
    Debug configuration for 256x256 dataset.
    """
    DEBUG = True
    DEBUG_MAX_BATCHES = 3
    IMAGE_SIZE = 128  # Even smaller crop for fast debugging
    NUM_WORKERS = 0


class Dataset1024DebugConfig(Dataset1024Config):
    """
    Debug configuration for 1024x1024 dataset.
    """
    DEBUG = True
    DEBUG_MAX_BATCHES = 3
    IMAGE_SIZE = 256  # Smaller crop for fast debugging
    NUM_WORKERS = 0

class DebugConfig(DefaultConfig):
    """
    Configuration for debugging - reduces training for fast tests.
    """
    DEBUG = True
    DEBUG_MAX_BATCHES = 3
    IMAGE_SIZE = 256  # Smaller images for faster debugging
    NUM_WORKERS = 0


# Dictionary for easy access to configurations
CONFIGS = {
    'default': DefaultConfig,
    'debug': DebugConfig,
    
    # Configurations for 256x256 dataset
    'dataset_256': Dataset256Config,
    'dataset_256_debug': Dataset256DebugConfig,
    
    # Configurations for 1024x1024 dataset
    'dataset_1024': Dataset1024Config,
    'dataset_1024_debug': Dataset1024DebugConfig,
}


def build_dataset_path(base_path, dataset_resolution):
    """
    Returns the base dataset path without modifications.
    The dataset is always in the same path regardless of the configured resolution.
    
    Args:
        base_path (str): Base path of the dataset
        dataset_resolution (int): Dataset resolution (256 or 1024) - used only for configuration
    
    Returns:
        str: Dataset path (always the base path)
    
    Example:
        >>> build_dataset_path('/path/to/dataset/', 256)
        '/path/to/dataset/'
        >>> build_dataset_path('/path/to/dataset/', 1024)
        '/path/to/dataset/'
    """
    # Always return the base path without modifications
    return base_path


def get_config(config_name='default'):
    """
    Returns the requested configuration.
    
    Args:
        config_name (str): Configuration name ('default', 'debug', 'high_performance', 'production')
    
    Returns:
        Config class: The requested configuration class
    """
    if config_name not in CONFIGS:
        available = ', '.join(CONFIGS.keys())
        raise ValueError(f"Configuration '{config_name}' not found. Available: {available}")
    
    return CONFIGS[config_name]


def print_config(config_class):
    """
    Print all configurations of a class.
    
    Args:
        config_class: The configuration class to print
    """
    print(f"=== CONFIGURATION: {config_class.__name__} ===")
    for attr in dir(config_class):
        if not attr.startswith('_') and attr.isupper():
            value = getattr(config_class, attr)
            print(f"{attr}: {value}")
    print("=" * 50)
