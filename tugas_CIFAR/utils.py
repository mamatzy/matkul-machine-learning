"""
Utility functions for CIFAR-10 Classification Project
"""
import os
import json
import torch
import numpy as np
from datetime import datetime


def create_directories():
    """Create necessary directories"""
    os.makedirs('./data', exist_ok=True)
    os.makedirs('./results', exist_ok=True)
    os.makedirs('./checkpoints', exist_ok=True)


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    print(f"Random seed set to {seed}")


def get_device(use_gpu=True):
    """
    Get device (GPU or CPU)
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model):
    """
    Count total number of trainable parameters in model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """
    Get model size in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    return total_size_mb


def format_time(seconds):
    """
    Format time in seconds to readable format
    """
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(secs)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(secs)}s"
    else:
        return f"{int(secs)}s"


def save_config(config_dict, filename='config_used.json'):
    """
    Save configuration to JSON file
    """
    with open(f'results/{filename}', 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f"Configuration saved: results/{filename}")


def log_experiment(experiment_info, filename='experiment_log.json'):
    """
    Log experiment details
    """
    log_path = f'results/{filename}'
    
    # Add timestamp
    experiment_info['timestamp'] = datetime.now().isoformat()
    
    # Load existing log or create new
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)
    else:
        log_data = {'experiments': []}
    
    log_data['experiments'].append(experiment_info)
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"Experiment logged: {log_path}")


def print_model_summary(model, model_name='Model'):
    """
    Print model summary
    """
    print(f"\n{'='*60}")
    print(f"{model_name} Summary")
    print(f"{'='*60}")
    
    total_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"\n{model}")
    print(f"{'='*60}\n")


def validate_checkpoint_exists(checkpoint_name):
    """
    Check if checkpoint exists
    """
    checkpoint_path = f'checkpoints/{checkpoint_name}'
    exists = os.path.exists(checkpoint_path)
    
    if exists:
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✓ Checkpoint found: {checkpoint_path} ({file_size:.2f} MB)")
    else:
        print(f"✗ Checkpoint not found: {checkpoint_path}")
    
    return exists


def load_experiment_log(filename='experiment_log.json'):
    """
    Load and display experiment log
    """
    log_path = f'results/{filename}'
    
    if not os.path.exists(log_path):
        print(f"No experiment log found at {log_path}")
        return None
    
    with open(log_path, 'r') as f:
        log_data = json.load(f)
    
    return log_data


def compare_experiments(filename='experiment_log.json'):
    """
    Compare all experiments in log
    """
    log_data = load_experiment_log(filename)
    
    if not log_data or 'experiments' not in log_data:
        print("No experiments to compare")
        return
    
    experiments = log_data['experiments']
    
    print(f"\n{'='*70}")
    print(f"Experiment Comparison ({len(experiments)} experiments)")
    print(f"{'='*70}\n")
    
    for i, exp in enumerate(experiments, 1):
        print(f"Experiment {i}:")
        print(f"  Timestamp: {exp.get('timestamp', 'N/A')}")
        print(f"  Best Accuracy: {exp.get('best_accuracy', 'N/A')}")
        print(f"  Training Time: {exp.get('training_time', 'N/A')}")
        print()


class ProgressTracker:
    """
    Track and display training progress
    """
    def __init__(self, total_epochs, print_interval=10):
        self.total_epochs = total_epochs
        self.print_interval = print_interval
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def update(self, epoch, train_loss, train_acc, val_loss, val_acc):
        """Update progress"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        
        if (epoch + 1) % self.print_interval == 0:
            self.print_status(epoch)
    
    def print_status(self, epoch):
        """Print current status"""
        idx = -1
        print(f"Epoch [{epoch+1}/{self.total_epochs}]")
        print(f"  Train Loss: {self.history['train_loss'][idx]:.4f}, "
              f"Train Acc: {self.history['train_acc'][idx]:.2f}%")
        print(f"  Val Loss: {self.history['val_loss'][idx]:.4f}, "
              f"Val Acc: {self.history['val_acc'][idx]:.2f}%")
    
    def get_best_accuracy(self):
        """Get best validation accuracy"""
        return max(self.history['val_acc']) if self.history['val_acc'] else 0
    
    def get_best_epoch(self):
        """Get epoch with best validation accuracy"""
        if not self.history['val_acc']:
            return 0
        return self.history['val_acc'].index(max(self.history['val_acc']))


def summarize_results(results_dict):
    """
    Print summary of results
    """
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70 + "\n")
    
    # Sort by accuracy
    sorted_results = sorted(
        results_dict.items(),
        key=lambda x: x[1].get('overall_accuracy', 0),
        reverse=True
    )
    
    print(f"{'Model':<20} {'Accuracy':<12} {'Time (s)':<12} {'Size (MB)':<12}")
    print("-" * 70)
    
    for name, metrics in sorted_results:
        acc = metrics.get('overall_accuracy', 0)
        time = metrics.get('total_time', 0)
        size = metrics.get('model_size_mb', 0)
        
        print(f"{name:<20} {acc:>10.2f}% {time:>10.2f}s {size:>10.2f} MB")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    # Test utility functions
    create_directories()
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
