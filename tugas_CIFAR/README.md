# CIFAR-10 Image Classification: Architecture and Pooling Strategy Comparison

This project implements and compares two deep learning architectures (VGG and ResNet) with different pooling strategies (Max vs Average Pooling) on the CIFAR-10 dataset.

## Overview

The exercise compares:
- **VGG** with Max Pooling
- **VGG** with Average Pooling  
- **ResNet** with Max Pooling
- **ResNet** with Average Pooling

### CIFAR-10 Dataset
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 32×32 pixels in RGB

## Project Structure

```
tugas_CIFAR/
├── main.py                  # Main orchestrator script
├── train.py                 # Training pipeline
├── evaluate.py              # Evaluation and visualization
├── models.py                # Model architectures (VGG, ResNet)
├── data_loader.py           # Data loading and preprocessing
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── data/                   # Downloaded CIFAR-10 dataset (auto-created)
└── results/               # Output directory (auto-created)
    ├── training_curves.png
    ├── accuracy_comparison.png
    ├── computational_metrics.png
    ├── confusion_matrix_*.png
    └── comparison_report.json
```

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

Alternatively, install manually:
```bash
pip install torch torchvision numpy matplotlib scikit-learn seaborn
```

## Usage

### Quick Start - Run Everything

Run the complete pipeline (data loading, training, evaluation, and visualization):
```bash
python main.py
```

This will:
1. Download CIFAR-10 dataset
2. Visualize sample images
3. Train all 4 model configurations (50 epochs each)
4. Evaluate models on test set
5. Generate comparison visualizations
6. Create detailed report

### Individual Scripts

**Option 1: Train Models Only**
```bash
python train.py
```
This trains all 4 models and saves the best weights for each.

**Option 2: Evaluate Pre-trained Models**
```bash
python evaluate.py
```
This evaluates pre-trained models (requires trained weights).

**Option 3: Data Exploration**
```bash
python data_loader.py
```
This downloads the data and visualizes sample images.

**Option 4: Model Architecture Testing**
```bash
python models.py
```
This tests the model implementations.

## Model Architectures

### VGG (Visual Geometry Group)
- Deep convolutional network with small 3×3 filters
- 4 blocks with increasing channel depth (64 → 512)
- 2 fully connected layers for classification
- Total parameters: ~20 million

```
Structure:
Block 1: Conv(3,64) → Conv(64,64) → Pooling
Block 2: Conv(64,128) → Conv(128,128) → Pooling
Block 3: Conv(128,256) × 3 → Pooling
Block 4: Conv(256,512) × 3 → Pooling
FC: 512 → 512 → 10
```

### ResNet (Residual Network)
- Uses residual connections to enable deeper networks
- 4 residual blocks with skip connections
- Batch normalization for stable training
- Total parameters: ~11 million (ResNet-18)

```
Structure:
Conv(3,64) → Pooling
Layer1: ResBlock(64,64) × 2
Layer2: ResBlock(64,128) × 2 + stride-2
Layer3: ResBlock(128,256) × 2 + stride-2
Layer4: ResBlock(256,512) × 2 + stride-2
GlobalAvgPool → FC(512,10)
```

## Pooling Strategies

### Max Pooling
- Takes maximum value from each pooling window
- Better for preserving sharp features
- Generally leads to higher accuracy
- More selective feature extraction

### Average Pooling
- Computes average of all values in pooling window
- Smoother, more distributed feature extraction
- Might lose important sharp features
- Generally leads to lower accuracy but sometimes better generalization

## Training Configuration

- **Optimizer**: SGD with momentum=0.9
- **Learning Rate**: 0.1 (with cosine annealing)
- **Weight Decay**: 5e-4 (L2 regularization)
- **Batch Size**: 32
- **Epochs**: 50
- **Data Augmentation**: Random crops, horizontal flips

## Expected Results

### Typical Accuracy Results (after 50 epochs)
- **VGG-MaxPool**: ~73-75%
- **VGG-AvgPool**: ~71-73%
- **ResNet-MaxPool**: ~75-77%
- **ResNet-AvgPool**: ~73-75%

**Note**: Actual results may vary based on random initialization and GPU/CPU variance.

### Key Observations

1. **Max Pooling vs Average Pooling**
   - Max pooling generally outperforms average pooling by 2-4%
   - Max pooling better preserves discriminative features
   - Average pooling may lead to smoother but less informative features

2. **VGG vs ResNet**
   - ResNet achieves higher accuracy with fewer parameters
   - ResNet is more stable during training due to skip connections
   - VGG may overfit on CIFAR-10 despite regularization
   - ResNet trains faster than VGG

3. **Computational Cost**
   - VGG: ~20M parameters, ~50-60s per epoch
   - ResNet: ~11M parameters, ~30-40s per epoch
   - ResNet is more efficient despite competitive accuracy

## Output Visualizations

### 1. training_curves.png
Four subplots showing:
- Training loss over epochs
- Test loss over epochs
- Training accuracy over epochs
- Test accuracy over epochs

### 2. accuracy_comparison.png
Two subplots:
- Overall test accuracy comparison (bar chart)
- Per-class accuracy comparison (grouped bar chart)

### 3. computational_metrics.png
Two subplots:
- Model size (in MB)
- Training time (in seconds)

### 4. confusion_matrix_*.png
4 confusion matrices (one per model) showing:
- True vs predicted class distribution
- Diagonal = correct predictions
- Off-diagonal = misclassifications

### 5. comparison_report.json
Structured data containing:
- Overall accuracy for each model
- Best/worst performing classes
- Model size and parameter count
- Training time

## Tips for Running

### If GPU is Available
- The code automatically uses GPU if available
- Training will be significantly faster (~5-10x speedup)

### If Out of Memory
- Reduce batch size: `python train.py` → modify `batch_size=16`
- Reduce number of epochs

### To Stop Training Early
- Press Ctrl+C to interrupt
- The best model weights up to that point are already saved

### To Resume Training
- Weights are automatically saved
- Running training again will overwrite previous weights
- Modify code to load and continue from checkpoint

## File Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Orchestrates entire pipeline |
| `train.py` | Training loop and Trainer class |
| `evaluate.py` | Model evaluation and visualization |
| `models.py` | VGG and ResNet implementations |
| `data_loader.py` | CIFAR-10 dataset loading |
| `requirements.txt` | Python dependencies |

## Troubleshooting

**Error: "ModuleNotFoundError: No module named 'torch'"**
- Solution: Run `pip install -r requirements.txt`

**Error: "CUDA out of memory"**
- Solution: Reduce batch size or use CPU: Set `device = torch.device('cpu')`

**Error: "No such file or directory: ./data"**
- Solution: Data directory is auto-created. Just run the script.

**Models not loading in evaluate.py**
- Solution: Train models first using `train.py`

## Academic References

1. **VGG Networks**: Simonyan & Zisserman (2014) - "Very Deep Convolutional Networks for Large-Scale Image Recognition"

2. **ResNet**: He et al. (2015) - "Deep Residual Learning for Image Recognition"

3. **CIFAR-10**: Krizhevsky & Hinton (2009) - "Learning Multiple Layers of Features from Tiny Images"

4. **Pooling Strategies**: Graham (2014) - "Fractional Max-Pooling"

## Author Notes

This project demonstrates:
- Deep learning best practices
- Architecture design and optimization
- Systematic model comparison methodology
- Comprehensive evaluation and visualization
- Professional code organization

## License

This project is for educational purposes.

## Contact & Support

For questions or issues, refer to the code comments and documentation within each file.

---

**Last Updated**: May 2026  
**Python Version**: 3.7+  
**PyTorch Version**: 1.9.0+
