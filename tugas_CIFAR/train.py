"""
Training script for CIFAR-10 models
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from data_loader import load_cifar10_data
from models import create_model


class Trainer:
    def __init__(self, model, device, model_name='model'):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1, 
                                   momentum=0.9, weight_decay=5e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=200)
        
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_losses = []
        self.total_time = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def test(self, test_loader):
        """
        Test the model
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = total_loss / len(test_loader)
        epoch_accuracy = 100 * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader, test_loader, num_epochs=50):
        """
        Main training loop
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        best_accuracy = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Test
            test_loss, test_acc = self.test(test_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                self.save_model(f"best_{self.model_name}.pth")
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
                print()
        
        self.total_time = time.time() - start_time
        
        print(f"{'='*60}")
        print(f"Training completed! Total time: {self.total_time:.2f}s")
        print(f"Best Test Accuracy: {best_accuracy:.2f}%")
        print(f"{'='*60}\n")
        
        return self.train_losses, self.train_accuracies, self.test_losses, self.test_accuracies
    
    def save_model(self, filename):
        """
        Save model to file
        """
        torch.save(self.model.state_dict(), filename)
        print(f"  Model saved: {filename}")
    
    def load_model(self, filename):
        """
        Load model from file
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))
        print(f"  Model loaded: {filename}")


def train_all_models(num_epochs=50, batch_size=32):
    """
    Train all 4 model configurations
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    train_loader, test_loader, classes = load_cifar10_data(batch_size=batch_size)
    print(f"Data loaded! Classes: {len(classes)}\n")
    
    # Models to train
    models_config = [
        ('vgg', 'max', 'VGG-MaxPool'),
        ('vgg', 'avg', 'VGG-AvgPool'),
        ('resnet', 'max', 'ResNet-MaxPool'),
        ('resnet', 'avg', 'ResNet-AvgPool'),
    ]
    
    trainers = []
    results = {}
    
    for arch, pooling, name in models_config:
        print(f"\nCreating {name}...")
        model = create_model(architecture=arch, pooling_type=pooling, num_classes=10)
        trainer = Trainer(model, device, model_name=name)
        
        # Train
        trainer.train(train_loader, test_loader, num_epochs=num_epochs)
        
        # Store results
        results[name] = {
            'trainer': trainer,
            'train_losses': trainer.train_losses,
            'train_accuracies': trainer.train_accuracies,
            'test_losses': trainer.test_losses,
            'test_accuracies': trainer.test_accuracies,
            'total_time': trainer.total_time,
            'best_accuracy': max(trainer.test_accuracies)
        }
        trainers.append(trainer)
    
    return results, trainers, train_loader, test_loader, classes


if __name__ == "__main__":
    results, trainers, train_loader, test_loader, classes = train_all_models(num_epochs=50, batch_size=32)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  Best Test Accuracy: {result['best_accuracy']:.2f}%")
        print(f"  Training Time: {result['total_time']:.2f}s")
