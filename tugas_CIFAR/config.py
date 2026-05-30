"""
Configuration file for CIFAR-10 Classification Project
Modify these parameters to customize the training and evaluation
"""

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

# Number of training epochs
NUM_EPOCHS = 50

# Batch size for training and testing
BATCH_SIZE = 32

# Number of workers for data loading
NUM_WORKERS = 2

# Learning rate
LEARNING_RATE = 0.1

# Momentum for SGD optimizer
MOMENTUM = 0.9

# Weight decay (L2 regularization)
WEIGHT_DECAY = 5e-4

# Temperature for learning rate scheduler (cosine annealing)
LR_SCHEDULER_T_MAX = 200

# Data augmentation: random crop padding
CROP_PADDING = 4

# Dropout probability in classifier
DROPOUT_PROB = 0.5

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# ResNet depth (18 or 34)
RESNET_DEPTH = 18

# Number of output classes
NUM_CLASSES = 10

# ============================================================================
# DATASET PARAMETERS
# ============================================================================

# Data directory (relative path)
DATA_DIR = './data'

# CIFAR-10 class names
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Normalization statistics for CIFAR-10
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# ============================================================================
# OUTPUT PARAMETERS
# ============================================================================

# Output directory for results
OUTPUT_DIR = './results'

# Save best model weights
SAVE_BEST_MODEL = True

# Model checkpoint directory
CHECKPOINT_DIR = './checkpoints'

# ============================================================================
# VISUALIZATION PARAMETERS
# ============================================================================

# DPI for saved figures
FIGURE_DPI = 300

# Figure format (png, jpg, pdf)
FIGURE_FORMAT = 'png'

# Number of sample images to visualize
NUM_SAMPLES_TO_VIS = 10

# ============================================================================
# DEVICE PARAMETERS
# ============================================================================

# Use GPU if available (True/False)
USE_GPU = True

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

MODEL_CONFIGS = [
    {
        'architecture': 'vgg',
        'pooling_type': 'max',
        'name': 'VGG-MaxPool'
    },
    {
        'architecture': 'vgg',
        'pooling_type': 'avg',
        'name': 'VGG-AvgPool'
    },
    {
        'architecture': 'resnet',
        'pooling_type': 'max',
        'name': 'ResNet-MaxPool'
    },
    {
        'architecture': 'resnet',
        'pooling_type': 'avg',
        'name': 'ResNet-AvgPool'
    },
]

# ============================================================================
# LOGGING PARAMETERS
# ============================================================================

# Print interval (print stats every N epochs)
PRINT_INTERVAL = 10

# Verbose output (True/False)
VERBOSE = True

# ============================================================================
# SEED FOR REPRODUCIBILITY
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42


if __name__ == "__main__":
    print("Configuration loaded!")
    print(f"Num Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Num Classes: {NUM_CLASSES}")
    print(f"Model Configs: {len(MODEL_CONFIGS)}")
