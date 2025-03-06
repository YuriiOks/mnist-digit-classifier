import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import logging

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # No gradients needed for evaluation
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Track statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def get_confusion_matrix(model, data_loader, device):
    """
    Generate confusion matrix for the model.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        
    Returns:
        numpy.ndarray: Confusion matrix
    """
    model.eval()
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Generating Confusion Matrix'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predicted)
    
    return cm

def plot_confusion_matrix(cm, class_names=None):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def visualize_model_predictions(model, data_loader, device, num_images=25):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with samples
        device: Device to run inference on
        num_images: Number of images to visualize
    """
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(15, 12))
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(images.size(0)):
                images_so_far += 1
                ax = plt.subplot(5, 5, images_so_far)
                ax.axis('off')
                ax.set_title(f'Pred: {preds[j]}, True: {labels[j]}')
                
                # Reverse normalization for display
                mean = torch.tensor([0.1307]).view(1, 1, 1)
                std = torch.tensor([0.3081]).view(1, 1, 1)
                img = images[j].cpu() * std + mean
                
                ax.imshow(img.squeeze().numpy(), cmap='gray')
                
                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('model_predictions.png')
                    plt.show()
                    return
    
    plt.tight_layout()
    plt.savefig('model_predictions.png')
    plt.show()
