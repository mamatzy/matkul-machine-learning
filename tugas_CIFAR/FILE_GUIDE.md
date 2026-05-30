# Project File Structure and Guide

## Quick Navigation

```
tugas_CIFAR/
├── 📋 Documentation
│   ├── README.md              # Complete project documentation
│   └── FILE_GUIDE.md          # This file
├── 🚀 Quick Start
│   ├── main.py                # Main orchestrator - START HERE
│   └── quickstart.py           # Interactive quick start menu
├── 🎓 Core Modules
│   ├── train.py               # Training pipeline
│   ├── evaluate.py            # Evaluation and visualization
│   ├── models.py              # Model architectures (VGG, ResNet)
│   └── data_loader.py         # Data loading and preprocessing
├── ⚙️ Configuration & Utils
│   ├── config.py              # Configuration parameters
│   ├── utils.py               # Helper functions
│   └── requirements.txt       # Python dependencies
└── 📁 Runtime Directories (auto-created)
    ├── data/                  # Downloaded CIFAR-10 dataset
    ├── results/               # Output visualizations & reports
    └── checkpoints/           # Saved model weights
```

## File Descriptions

### 📋 Documentation Files

#### `README.md` (Start Here!)
- **Purpose**: Complete project documentation
- **Contains**: 
  - Project overview and objectives
  - Installation instructions
  - Usage guide for all scripts
  - Expected results and benchmarks
  - Model architecture explanations
  - Troubleshooting guide
- **When to read**: First thing! Read this to understand the project

#### `FILE_GUIDE.md` (This File)
- **Purpose**: Navigation and file descriptions
- **When to use**: When you need to understand what each file does

---

### 🚀 Quick Start Files

#### `main.py`
- **Purpose**: Main orchestrator script - runs complete pipeline
- **What it does**:
  1. Loads CIFAR-10 dataset
  2. Trains all 4 models (50 epochs each)
  3. Evaluates models
  4. Generates visualizations
  5. Creates comparison report
- **Run**: `python main.py`
- **Time**: 1-2 hours (CPU), 15-30 minutes (GPU)
- **Output**: Multiple PNG files and JSON report in `results/`
- **Best for**: Getting complete results in one go

#### `quickstart.py`
- **Purpose**: Interactive menu-driven quick start
- **What it does**:
  - Provides interactive menu
  - Checks requirements
  - Offers various options:
    1. Train models
    2. Evaluate pre-trained models
    3. Visualize data
    4. Run complete pipeline
    5. Custom training with parameters
- **Run**: `python quickstart.py`
- **Best for**: First-time users or when you want flexibility

---

### 🎓 Core Module Files

#### `train.py`
- **Purpose**: Training pipeline and Trainer class
- **Main Classes**:
  - `Trainer`: Handles training loop for a single model
    - Methods: `train_epoch()`, `test()`, `train()`, `save_model()`
- **Main Functions**:
  - `train_all_models()`: Train all 4 model configurations
- **Run**: `python train.py`
- **Output**: Saved model weights in current directory
- **Best for**: When you only want to train models

**Key Variables**:
- `train_losses`: Training loss per epoch
- `train_accuracies`: Training accuracy per epoch
- `test_losses`: Test loss per epoch
- `test_accuracies`: Test accuracy per epoch

#### `evaluate.py`
- **Purpose**: Model evaluation and visualization
- **Main Classes**:
  - `ModelEvaluator`: Evaluates single model
    - Methods: `evaluate()`, `print_report()`, `plot_confusion_matrix()`
- **Main Functions**:
  - `compare_models()`: Evaluate all models
  - `plot_training_curves()`: Plot loss/accuracy curves
  - `plot_accuracy_comparison()`: Compare accuracies
  - `plot_computational_metrics()`: Compare size and time
  - `generate_comparison_report()`: Create JSON report
- **Run**: `python evaluate.py`
- **Prerequisites**: Trained model weights files
- **Output**: PNG files and JSON report
- **Best for**: Analyzing pre-trained models

#### `models.py`
- **Purpose**: Model architecture implementations
- **Main Classes**:
  - `VGGBase`: VGG architecture with customizable pooling
  - `ResNetBlock`: Residual block building block
  - `ResNetBase`: ResNet architecture with customizable pooling
- **Main Functions**:
  - `create_model()`: Factory function to create models
- **Run**: `python models.py` (tests models)
- **Usage**: 
  ```python
  from models import create_model
  model = create_model('vgg', 'max')  # VGG with Max Pooling
  model = create_model('resnet', 'avg')  # ResNet with Avg Pooling
  ```
- **Best for**: Understanding model architectures

#### `data_loader.py`
- **Purpose**: CIFAR-10 data loading and preprocessing
- **Main Functions**:
  - `load_cifar10_data()`: Load train/test loaders
  - `visualize_samples()`: Visualize sample images
- **Run**: `python data_loader.py`
- **Output**: CIFAR-10 dataset in `data/` folder
- **Features**:
  - Automatic download
  - Data augmentation (crops, flips)
  - Normalization
- **Best for**: Understanding data pipeline

---

### ⚙️ Configuration & Utility Files

#### `config.py`
- **Purpose**: Centralized configuration parameters
- **Contains**:
  - Training parameters (epochs, batch_size, learning_rate, etc.)
  - Model parameters (architecture options)
  - Dataset parameters (class names, normalization stats)
  - Output parameters (directories, formats)
  - Device parameters (GPU/CPU selection)
- **How to use**: Modify values in this file instead of editing code
- **Example**:
  ```python
  from config import NUM_EPOCHS, BATCH_SIZE, NUM_CLASSES
  ```
- **Best for**: Quick parameter adjustments without code changes

#### `utils.py`
- **Purpose**: Helper functions and utilities
- **Main Functions**:
  - `create_directories()`: Create output directories
  - `set_seed()`: Set random seeds for reproducibility
  - `get_device()`: Get GPU/CPU device
  - `count_parameters()`: Count model parameters
  - `get_model_size_mb()`: Get model size
  - `format_time()`: Format time nicely
  - `print_model_summary()`: Print model info
  - `summarize_results()`: Print results summary
- **Main Classes**:
  - `ProgressTracker`: Track training progress
- **Run**: `python utils.py` (tests functions)
- **Best for**: Reusable functionality across modules

#### `requirements.txt`
- **Purpose**: Python package dependencies
- **Contents**:
  ```
  torch>=1.9.0
  torchvision>=0.10.0
  numpy>=1.19.0
  matplotlib>=3.3.0
  scikit-learn>=0.24.0
  seaborn>=0.11.0
  ```
- **Install**: `pip install -r requirements.txt`
- **Best for**: Setting up environment

---

## Workflow Guide

### Beginner: First-time user
1. Read `README.md` for overview
2. Run `python quickstart.py` to explore options interactively
3. Choose option 4 (Complete Pipeline) to run everything
4. Review generated visualizations in `results/`

### Intermediate: Train custom models
1. Modify parameters in `config.py`
2. Run `python train.py` to train models
3. Review saved weights in current directory
4. Run `python evaluate.py` to evaluate and visualize

### Advanced: Use individual components
1. Use `data_loader.py` to create custom data loaders
2. Import models from `models.py` and create custom architectures
3. Use `train.py` Trainer class for custom training loops
4. Use `evaluate.py` evaluation functions for analysis

---

## Output Files Generated

### During Training
- `best_VGG-MaxPool.pth` - Trained VGG with max pooling
- `best_VGG-AvgPool.pth` - Trained VGG with avg pooling
- `best_ResNet-MaxPool.pth` - Trained ResNet with max pooling
- `best_ResNet-AvgPool.pth` - Trained ResNet with avg pooling

### In `results/` directory
- `training_curves.png` - Loss and accuracy curves
- `accuracy_comparison.png` - Accuracy bar charts
- `computational_metrics.png` - Size and time comparison
- `confusion_matrix_VGG-MaxPool.png` - Confusion matrix
- `confusion_matrix_VGG-AvgPool.png` - Confusion matrix
- `confusion_matrix_ResNet-MaxPool.png` - Confusion matrix
- `confusion_matrix_ResNet-AvgPool.png` - Confusion matrix
- `comparison_report.json` - Detailed metrics in JSON

### In `data/` directory
- `cifar-10-batches-py/` - Downloaded CIFAR-10 dataset

---

## Key Concepts

### VGG (Visual Geometry Group)
- Deep CNN with small 3×3 filters
- 4 convolutional blocks
- Two fully connected layers
- ~20M parameters
- Good accuracy but slower training

### ResNet (Residual Networks)
- Uses skip connections (residual blocks)
- Enables training of deeper networks
- ~11M parameters (ResNet-18)
- Better accuracy with faster training

### Max Pooling vs Average Pooling
- **Max**: Takes maximum value, sharper features, better accuracy
- **Average**: Averages values, smoother features, sometimes better generalization

### CIFAR-10 Dataset
- 60,000 images (50K train, 10K test)
- 10 classes
- 32×32 RGB images
- Standard benchmark for image classification

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'torch'" | Run `pip install -r requirements.txt` |
| "CUDA out of memory" | Reduce BATCH_SIZE in config.py |
| "Models not loading" | Train models first with train.py |
| "Permission denied" | Run `chmod +x quickstart.py` on Linux/Mac |
| "Download error" | Check internet connection, re-run data_loader.py |

---

## Tips for Best Results

1. **First run**: Use quickstart.py for guided experience
2. **Quick experiment**: Use 10 epochs in config.py
3. **GPU training**: 10-15x faster than CPU
4. **Reproducibility**: Use utils.set_seed() with same seed
5. **Save results**: Copy `results/` folder before new runs

---

## Performance Notes

- **VGG**: Slower but good baseline (50-60s per epoch on CPU)
- **ResNet**: Faster with better accuracy (30-40s per epoch on CPU)
- **GPU**: 5-10x faster overall
- **50 epochs**: ~1-2 hours on CPU, ~15-30 min on GPU

---

## Questions or Issues?

1. Check README.md for detailed documentation
2. Review code comments in relevant module
3. Check output messages and error logs
4. Verify requirements are installed

---

**Happy Learning! 🚀**

For more information, see README.md
