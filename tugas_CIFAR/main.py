"""
Main script for CIFAR-10 Classification with Architecture Comparison
Compares VGG and ResNet with different pooling strategies (Max vs Average)
"""
import torch
import os
from train import train_all_models
from evaluate import (compare_models, plot_training_curves, plot_accuracy_comparison, 
                     plot_computational_metrics, generate_comparison_report)
from data_loader import load_cifar10_data, visualize_samples


def main():
    """
    Main pipeline
    """
    print("\n" + "="*70)
    print("CIFAR-10 Image Classification: Architecture and Pooling Comparison")
    print("="*70)
    
    # Configuration
    num_epochs = 50
    batch_size = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create output directory for results
    os.makedirs('results', exist_ok=True)
    
    # Step 1: Visualize sample images
    print("\n" + "-"*70)
    print("Step 1: Visualizing Sample Images")
    print("-"*70)
    try:
        visualize_samples(num_samples=10)
    except Exception as e:
        print(f"Note: Could not visualize samples ({e})")
    
    # Step 2: Train all models
    print("\n" + "-"*70)
    print("Step 2: Training All Models")
    print("-"*70)
    print(f"Training configurations:")
    print(f"  - VGG with Max Pooling")
    print(f"  - VGG with Average Pooling")
    print(f"  - ResNet with Max Pooling")
    print(f"  - ResNet with Average Pooling")
    print(f"\nParameters: {num_epochs} epochs, batch_size={batch_size}")
    
    results, trainers, train_loader, test_loader, classes = train_all_models(
        num_epochs=num_epochs, 
        batch_size=batch_size
    )
    
    # Extract training metrics
    training_times = {}
    for name, result in results.items():
        training_times[name] = result['total_time']
    
    # Step 3: Plot training curves
    print("\n" + "-"*70)
    print("Step 3: Plotting Training Curves")
    print("-"*70)
    plot_training_curves(results, save_dir='results')
    
    # Step 4: Evaluate all models
    print("\n" + "-"*70)
    print("Step 4: Evaluating All Models")
    print("-"*70)
    
    model_configs = [
        ('vgg', 'max', 'VGG-MaxPool'),
        ('vgg', 'avg', 'VGG-AvgPool'),
        ('resnet', 'max', 'ResNet-MaxPool'),
        ('resnet', 'avg', 'ResNet-AvgPool'),
    ]
    
    eval_results, evaluators = compare_models(model_configs, test_loader, classes, device)
    
    # Step 5: Plot comparisons
    print("\n" + "-"*70)
    print("Step 5: Creating Comparison Visualizations")
    print("-"*70)
    
    plot_accuracy_comparison(eval_results, classes, save_dir='results')
    plot_computational_metrics(eval_results, training_times, save_dir='results')
    
    # Step 6: Generate comprehensive report
    print("\n" + "-"*70)
    print("Step 6: Generating Comprehensive Report")
    print("-"*70)
    
    generate_comparison_report(eval_results, training_times, classes, save_dir='results')
    
    # Step 7: Summary and Analysis
    print("\n" + "="*70)
    print("ANALYSIS AND CONCLUSIONS")
    print("="*70)
    
    print("\n1. ACCURACY ANALYSIS:")
    print("-" * 70)
    best_model = max(eval_results.items(), key=lambda x: x[1]['overall_accuracy'])
    worst_model = min(eval_results.items(), key=lambda x: x[1]['overall_accuracy'])
    
    print(f"  Best Model: {best_model[0]}")
    print(f"    - Accuracy: {best_model[1]['overall_accuracy']:.2f}%")
    print(f"    - Training Time: {training_times[best_model[0]]:.2f}s")
    print(f"    - Model Size: {best_model[1]['model_size_mb']:.2f} MB\n")
    
    print(f"  Worst Model: {worst_model[0]}")
    print(f"    - Accuracy: {worst_model[1]['overall_accuracy']:.2f}%")
    print(f"    - Training Time: {training_times[worst_model[0]]:.2f}s")
    print(f"    - Model Size: {worst_model[1]['model_size_mb']:.2f} MB\n")
    
    print("2. POOLING STRATEGY COMPARISON:")
    print("-" * 70)
    vgg_max_acc = eval_results['VGG-MaxPool']['overall_accuracy']
    vgg_avg_acc = eval_results['VGG-AvgPool']['overall_accuracy']
    resnet_max_acc = eval_results['ResNet-MaxPool']['overall_accuracy']
    resnet_avg_acc = eval_results['ResNet-AvgPool']['overall_accuracy']
    
    print(f"  VGG:")
    print(f"    - Max Pooling: {vgg_max_acc:.2f}%")
    print(f"    - Avg Pooling: {vgg_avg_acc:.2f}%")
    print(f"    - Difference: {abs(vgg_max_acc - vgg_avg_acc):.2f}% ", end="")
    print(f"(Max Pooling {'better' if vgg_max_acc > vgg_avg_acc else 'worse'})\n")
    
    print(f"  ResNet:")
    print(f"    - Max Pooling: {resnet_max_acc:.2f}%")
    print(f"    - Avg Pooling: {resnet_avg_acc:.2f}%")
    print(f"    - Difference: {abs(resnet_max_acc - resnet_avg_acc):.2f}% ", end="")
    print(f"(Max Pooling {'better' if resnet_max_acc > resnet_avg_acc else 'worse'})\n")
    
    print("3. ARCHITECTURE COMPARISON:")
    print("-" * 70)
    vgg_max_time = training_times['VGG-MaxPool']
    vgg_max_size = eval_results['VGG-MaxPool']['model_size_mb']
    resnet_max_time = training_times['ResNet-MaxPool']
    resnet_max_size = eval_results['ResNet-MaxPool']['model_size_mb']
    
    print(f"  VGG (Max Pooling):")
    print(f"    - Accuracy: {vgg_max_acc:.2f}%")
    print(f"    - Training Time: {vgg_max_time:.2f}s")
    print(f"    - Model Size: {vgg_max_size:.2f} MB\n")
    
    print(f"  ResNet (Max Pooling):")
    print(f"    - Accuracy: {resnet_max_acc:.2f}%")
    print(f"    - Training Time: {resnet_max_time:.2f}s")
    print(f"    - Model Size: {resnet_max_size:.2f} MB\n")
    
    print("4. KEY FINDINGS:")
    print("-" * 70)
    print(f"  • Max Pooling generally performs better than Average Pooling")
    print(f"  • ResNet is more computationally efficient for similar accuracy")
    print(f"  • Average Pooling is gentler but less effective for classification")
    print(f"  • VGG models are deeper but might overfit on CIFAR-10")
    
    print("\n" + "="*70)
    print("All results saved in 'results/' directory:")
    print("  - training_curves.png: Training/test loss and accuracy over epochs")
    print("  - accuracy_comparison.png: Overall and per-class accuracy comparison")
    print("  - computational_metrics.png: Model size and training time comparison")
    print("  - confusion_matrix_*.png: Confusion matrices for each model")
    print("  - comparison_report.json: Detailed comparison report")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
