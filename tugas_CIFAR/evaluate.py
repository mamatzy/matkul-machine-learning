"""
Evaluation and Visualization script for model comparison
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from data_loader import load_cifar10_data
from models import create_model
import json


class ModelEvaluator:
    def __init__(self, model, test_loader, classes, device, model_name='model'):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.classes = classes
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        
        self.predictions = None
        self.ground_truth = None
        self.accuracies_per_class = None
        self.overall_accuracy = None
        self.confusion_mat = None
        self.model_size = self._calculate_model_size()
    
    def _calculate_model_size(self):
        """
        Calculate model size in MB
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        total_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
        return total_size_mb, total_params
    
    def evaluate(self):
        """
        Full evaluation of the model
        """
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        self.predictions = np.array(all_predictions)
        self.ground_truth = np.array(all_labels)
        
        # Calculate overall accuracy
        self.overall_accuracy = np.mean(self.predictions == self.ground_truth) * 100
        
        # Calculate per-class accuracy
        self.accuracies_per_class = []
        for i in range(len(self.classes)):
            mask = self.ground_truth == i
            if mask.sum() > 0:
                acc = np.mean(self.predictions[mask] == self.ground_truth[mask]) * 100
                self.accuracies_per_class.append(acc)
            else:
                self.accuracies_per_class.append(0)
        
        # Calculate confusion matrix
        self.confusion_mat = confusion_matrix(self.ground_truth, self.predictions)
        
        return self.overall_accuracy, self.accuracies_per_class
    
    def print_report(self):
        """
        Print evaluation report
        """
        print(f"\n{'='*60}")
        print(f"Evaluation Report: {self.model_name}")
        print(f"{'='*60}\n")
        
        print(f"Overall Accuracy: {self.overall_accuracy:.2f}%")
        print(f"Model Size: {self.model_size[0]:.2f} MB ({self.model_size[1]:,} parameters)\n")
        
        print("Per-Class Accuracy:")
        for i, acc in enumerate(self.accuracies_per_class):
            print(f"  {self.classes[i]:10s}: {acc:.2f}%")
        
        print(f"\n{'='*60}\n")
    
    def plot_confusion_matrix(self, filename=None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.confusion_mat, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.classes, yticklabels=self.classes)
        plt.title(f'Confusion Matrix - {self.model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved: {filename}")
        plt.close()


def compare_models(model_configs, test_loader, classes, device):
    """
    Compare multiple models
    
    Args:
        model_configs: List of tuples (architecture, pooling, name)
        test_loader: Test data loader
        classes: Class names
        device: Device to use
    
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    evaluators = []
    
    for arch, pooling, name in model_configs:
        print(f"\nEvaluating {name}...")
        
        # Create and load model
        model = create_model(architecture=arch, pooling_type=pooling, num_classes=10)
        
        # Try to load trained weights
        try:
            model.load_state_dict(torch.load(f"best_{name}.pth", map_location=device))
            print(f"  Loaded weights from best_{name}.pth")
        except:
            print(f"  WARNING: Could not load weights for {name}")
        
        # Evaluate
        evaluator = ModelEvaluator(model, test_loader, classes, device, name)
        accuracy, per_class_acc = evaluator.evaluate()
        
        evaluator.print_report()
        evaluator.plot_confusion_matrix(f"confusion_matrix_{name}.png")
        
        results[name] = {
            'overall_accuracy': accuracy,
            'per_class_accuracy': per_class_acc,
            'model_size_mb': evaluator.model_size[0],
            'num_parameters': evaluator.model_size[1],
            'confusion_matrix': evaluator.confusion_mat.tolist()
        }
        
        evaluators.append(evaluator)
    
    return results, evaluators


def plot_training_curves(results, save_dir='.'):
    """
    Plot training curves for all models
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training Loss
    ax = axes[0, 0]
    for name, result in results.items():
        ax.plot(result['train_losses'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Test Loss
    ax = axes[0, 1]
    for name, result in results.items():
        ax.plot(result['test_losses'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Loss')
    ax.set_title('Test Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Accuracy
    ax = axes[1, 0]
    for name, result in results.items():
        ax.plot(result['train_accuracies'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Accuracy (%)')
    ax.set_title('Training Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Test Accuracy
    ax = axes[1, 1]
    for name, result in results.items():
        ax.plot(result['test_accuracies'], label=name, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Test Accuracy Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_curves.png', dpi=300, bbox_inches='tight')
    print("Training curves saved: training_curves.png")
    plt.close()


def plot_accuracy_comparison(results, classes, save_dir='.'):
    """
    Plot accuracy comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Plot 1: Overall Accuracy
    ax = axes[0]
    model_names = list(results.keys())
    accuracies = [results[name]['overall_accuracy'] for name in model_names]
    
    bars = ax.bar(model_names, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Overall Test Accuracy Comparison')
    ax.set_ylim([0, 100])
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Per-class Accuracy
    ax = axes[1]
    x = np.arange(len(classes))
    width = 0.2
    
    for i, (name, result) in enumerate(results.items()):
        offset = (i - 1.5) * width
        ax.bar(x + offset, result['per_class_accuracy'], width, label=name)
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("Accuracy comparison saved: accuracy_comparison.png")
    plt.close()


def plot_computational_metrics(results, training_times, save_dir='.'):
    """
    Plot computational metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Model Size
    ax = axes[0]
    model_names = list(results.keys())
    model_sizes = [results[name]['model_size_mb'] for name in model_names]
    
    bars = ax.bar(model_names, model_sizes, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Training Time
    ax = axes[1]
    training_time_list = [training_times[name] for name in model_names]
    
    bars = ax.bar(model_names, training_time_list, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/computational_metrics.png', dpi=300, bbox_inches='tight')
    print("Computational metrics saved: computational_metrics.png")
    plt.close()


def generate_comparison_report(results, training_times, classes, save_dir='.'):
    """
    Generate a comprehensive comparison report
    """
    report = {
        'models': {}
    }
    
    for name, result in results.items():
        report['models'][name] = {
            'overall_accuracy': f"{result['overall_accuracy']:.2f}%",
            'best_class': classes[np.argmax(result['per_class_accuracy'])],
            'worst_class': classes[np.argmin(result['per_class_accuracy'])],
            'best_class_accuracy': f"{max(result['per_class_accuracy']):.2f}%",
            'worst_class_accuracy': f"{min(result['per_class_accuracy']):.2f}%",
            'model_size_mb': f"{result['model_size_mb']:.2f}",
            'num_parameters': f"{result['num_parameters']:,}",
            'training_time_seconds': f"{training_times[name]:.2f}"
        }
    
    # Save report
    with open(f'{save_dir}/comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\nComparison Report:")
    print("="*60)
    for name, metrics in report['models'].items():
        print(f"\n{name}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    
    print(f"\nFull report saved: comparison_report.json")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    _, test_loader, classes = load_cifar10_data(batch_size=32)
    
    # Model configurations
    model_configs = [
        ('vgg', 'max', 'VGG-MaxPool'),
        ('vgg', 'avg', 'VGG-AvgPool'),
        ('resnet', 'max', 'ResNet-MaxPool'),
        ('resnet', 'avg', 'ResNet-AvgPool'),
    ]
    
    # Evaluate all models
    results, evaluators = compare_models(model_configs, test_loader, classes, device)
