"""
Data Loading Module for CIFAR-10
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def load_cifar10_data(batch_size=32, num_workers=2):
    """
    Load CIFAR-10 dataset with preprocessing
    
    Args:
        batch_size: Batch size for training and testing
        num_workers: Number of workers for data loading
    
    Returns:
        train_loader, test_loader, classes
    """
    
    # Define transforms with data augmentation for training
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Test transforms without augmentation
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])
    
    # Load training and test datasets
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                     download=True, transform=train_transforms)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=test_transforms)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, test_loader, classes


def visualize_samples(num_samples=10):
    """
    Visualize sample images from CIFAR-10 dataset
    """
    _, test_loader, classes = load_cifar10_data(batch_size=num_samples)
    
    # Get a batch of images
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Denormalize
    images = images / 2 + 0.5
    images = images.numpy()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()
    
    for i in range(min(num_samples, images.shape[0])):
        img = np.transpose(images[i], (1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("Sample images saved as 'sample_images.png'")
    plt.close()


if __name__ == "__main__":
    # Test data loading
    train_loader, test_loader, classes = load_cifar10_data()
    print(f"Classes: {classes}")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Visualize samples
    visualize_samples()
