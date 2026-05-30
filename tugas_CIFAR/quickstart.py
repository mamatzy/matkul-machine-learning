#!/usr/bin/env python3
"""
Quick Start Script for CIFAR-10 Classification Project
Run this for an interactive setup and execution
"""
import os
import sys
import subprocess
import torch


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_section(text):
    """Print formatted section"""
    print("\n" + "-"*70)
    print(f"  {text}")
    print("-"*70)


def check_requirements():
    """Check if all required packages are installed"""
    print_section("Checking Requirements")
    
    required = ['torch', 'torchvision', 'numpy', 'matplotlib', 'sklearn', 'seaborn']
    missing = []
    
    for package in required:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing.append(package)
    
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        response = input("  Install missing packages? (y/n): ").lower()
        if response == 'y':
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("  Installation complete!")
        else:
            print("  Cannot proceed without required packages.")
            sys.exit(1)


def check_device():
    """Check available device"""
    print_section("Device Information")
    
    if torch.cuda.is_available():
        print(f"  GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print("  Status: Using GPU (Fast!)")
    else:
        print("  GPU: Not Available")
        print("  Status: Using CPU (Slower)")


def menu():
    """Display main menu"""
    print_header("CIFAR-10 Classification - Quick Start")
    
    print("\nSelect an option:")
    print("  1. Train all models (50 epochs)")
    print("  2. Evaluate pre-trained models")
    print("  3. Visualize sample images")
    print("  4. Run complete pipeline (train + evaluate)")
    print("  5. Customize parameters and train")
    print("  6. Exit")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    return choice


def train_models():
    """Train all models"""
    print_section("Training Models")
    print("Starting training... (This may take 30-45 minutes on CPU)\n")
    
    try:
        exec(open('train.py').read())
    except Exception as e:
        print(f"Error during training: {e}")


def evaluate_models():
    """Evaluate models"""
    print_section("Evaluating Models")
    print("Evaluating pre-trained models...\n")
    
    try:
        exec(open('evaluate.py').read())
    except Exception as e:
        print(f"Error during evaluation: {e}")


def visualize_data():
    """Visualize sample images"""
    print_section("Visualizing Data")
    print("Creating sample image visualization...\n")
    
    try:
        from data_loader import visualize_samples, load_cifar10_data
        print("Loading CIFAR-10 data...")
        visualize_samples(num_samples=10)
        print("Sample visualization saved!")
    except Exception as e:
        print(f"Error during visualization: {e}")


def run_complete_pipeline():
    """Run complete pipeline"""
    print_section("Running Complete Pipeline")
    print("This will: train models, evaluate, and generate visualizations")
    print("(This may take 1-2 hours on CPU)\n")
    
    try:
        exec(open('main.py').read())
    except Exception as e:
        print(f"Error during pipeline: {e}")


def custom_training():
    """Customize training parameters"""
    print_section("Custom Training")
    
    try:
        num_epochs = int(input("Number of epochs (default 50): ") or 50)
        batch_size = int(input("Batch size (default 32): ") or 32)
        
        print(f"\nTraining with {num_epochs} epochs and batch size {batch_size}...")
        print("(This may take a while)\n")
        
        from train import train_all_models
        results, trainers, train_loader, test_loader, classes = train_all_models(
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        
        print("\nTraining completed!")
        
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except Exception as e:
        print(f"Error during training: {e}")


def main():
    """Main function"""
    try:
        # Check requirements
        check_requirements()
        check_device()
        
        # Main loop
        while True:
            choice = menu()
            
            if choice == '1':
                train_models()
            elif choice == '2':
                evaluate_models()
            elif choice == '3':
                visualize_data()
            elif choice == '4':
                run_complete_pipeline()
            elif choice == '5':
                custom_training()
            elif choice == '6':
                print("\nGoodbye!")
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")
            
            # Ask if user wants to continue
            if choice in ['1', '2', '3', '4', '5']:
                again = input("\nContinue? (y/n): ").lower()
                if again != 'y':
                    print("Goodbye!")
                    break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
