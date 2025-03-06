import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate a model on a dataset
    
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
        pbar = tqdm(data_loader, desc='Evaluation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Calculate evaluation metrics
            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = correct / total
    
    return avg_loss, accuracy

def plot_training_history(history):
    """
    Plot training and validation loss/accuracy
    
    Args:
        history: Dictionary containing training history
    """
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def plot_confusion_matrix(model, data_loader, device, class_names=None):
    """
    Generate and plot a confusion matrix for the model
    
    Args:
        model: PyTorch model
        data_loader: PyTorch DataLoader
        device: Device to run evaluation on
        class_names: List of class names
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # Get predictions without computing gradients
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc='Creating Confusion Matrix'):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names if class_names else range(10),
                yticklabels=class_names if class_names else range(10))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()
