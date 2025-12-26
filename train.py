import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_model
import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns

class WSIDataset(Dataset):
    """
    Dataset class for WSI patches loaded from disk.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()

def load_dataset_from_folders(data_dir):
    """
    Load dataset from tumor/normal folder structure.
    
    Args:
        data_dir: Path to directory containing 'tumor' and 'normal' subdirectories
    
    Returns:
        image_paths: List of image file paths
        labels: List of labels (0=normal, 1=tumor)
    """
    data_path = Path(data_dir)
    tumor_dir = data_path / 'tumor'
    normal_dir = data_path / 'normal'
    
    image_paths = []
    labels = []
    
    # Load tumor images (label=1)
    if tumor_dir.exists():
        for img_file in tumor_dir.glob('*.[pj][np]g'):
            image_paths.append(str(img_file))
            labels.append(1)
    
    # Load normal images (label=0)
    if normal_dir.exists():
        for img_file in normal_dir.glob('*.[pj][np]g'):
            image_paths.append(str(img_file))
            labels.append(0)
    
    return image_paths, labels

def get_data_transforms(augment=False):
    """
    Get data transformations for ResNet-18 preprocessing.
    
    Args:
        augment: Whether to apply data augmentation (use for training)
    
    Returns:
        transforms.Compose: Data transformation pipeline
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of tumor class

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc, all_labels, all_predictions, all_probs

def calculate_metrics(y_true, y_pred, y_probs):
    """
    Calculate comprehensive evaluation metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.0
    }
    return metrics

def plot_training_history(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Plot training and validation loss/accuracy curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved training history plot to {save_dir}/training_history.png")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_dir):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Tumor'],
                yticklabels=['Normal', 'Tumor'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_dir}/confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_true, y_probs, save_dir):
    """
    Plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved ROC curve to {save_dir}/roc_curve.png")
    plt.close()

def plot_metrics_summary(metrics, save_dir):
    """
    Plot bar chart of all evaluation metrics.
    """
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metric_names, metric_values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'])
    plt.xlabel('Metrics', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Evaluation Metrics Summary', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Saved metrics summary to {save_dir}/metrics_summary.png")
    plt.close()

def main():
    # Configuration
    DATA_DIR = 'data'
    MODELS_DIR = 'models'
    RESULTS_DIR = 'results'
    BATCH_SIZE = 8  # Smaller batch size for CPU
    NUM_EPOCHS = 15  # Stage 1: Transfer learning
    LEARNING_RATE = 0.001
    VAL_SPLIT = 0.2  # 20% for validation
    PATIENCE = 5  # Early stopping patience
    
    # Create directories
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"STAGE 1: TRANSFER LEARNING (Frozen Backbone)")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    print(f"Batch size: {BATCH_SIZE} (optimized for CPU)")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Early stopping patience: {PATIENCE}")
    
    # Load dataset
    print(f"\nLoading dataset from {DATA_DIR}...")
    image_paths, labels = load_dataset_from_folders(DATA_DIR)
    
    if len(image_paths) == 0:
        print(f"\n❌ ERROR: No images found in {DATA_DIR}")
        print("Please run preprocessing first:")
        print("  python scripts/cluster_and_create_dataset.py --assign <cluster_id> --n 250")
        return
    
    print(f"Found {len(image_paths)} images")
    print(f"  - Tumor: {sum(labels)}")
    print(f"  - Normal: {len(labels) - sum(labels)}")
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=VAL_SPLIT, random_state=42, stratify=labels
    )
    
    print(f"\nTrain set: {len(train_paths)} images")
    print(f"Validation set: {len(val_paths)} images")
    
    # Create datasets and dataloaders
    train_transform = get_data_transforms(augment=True)
    val_transform = get_data_transforms(augment=False)
    
    train_dataset = WSIDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = WSIDataset(val_paths, val_labels, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    
    # Initialize model with FROZEN backbone (Stage 1: Transfer Learning)
    print(f"\nInitializing ResNet-18 with frozen backbone...")
    model = get_model(num_classes=2, pretrained=True, freeze_backbone=True)
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop with early stopping
    print(f"\n{'='*60}")
    print("Starting Training...")
    print(f"{'='*60}\n")
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_labels_all, val_preds_all, val_probs_all = validate_epoch(
            model, val_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model_stage1.pth'))
            print("✓ Best model saved!")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")
            
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
        
        print()
    
    # Load best model for final evaluation
    print(f"\n{'='*60}")
    print("Loading best model for final evaluation...")
    print(f"{'='*60}\n")
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, 'best_model_stage1.pth')))
    
    # Final evaluation on validation set
    val_loss, val_acc, val_labels_all, val_preds_all, val_probs_all = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(val_labels_all, val_preds_all, val_probs_all)
    
    print("\n📊 FINAL EVALUATION METRICS:")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 40)
    
    # Generate all plots
    print(f"\n📈 Generating evaluation plots...")
    plot_training_history(train_losses, val_losses, train_accs, val_accs, RESULTS_DIR)
    plot_confusion_matrix(val_labels_all, val_preds_all, RESULTS_DIR)
    plot_roc_curve(val_labels_all, val_probs_all, RESULTS_DIR)
    plot_metrics_summary(metrics, RESULTS_DIR)
    
    # Save metrics to file
    metrics_file = os.path.join(RESULTS_DIR, 'metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write("STAGE 1: TRANSFER LEARNING (Frozen Backbone)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  - Epochs trained: {len(train_losses)}\n")
        f.write(f"  - Batch size: {BATCH_SIZE}\n")
        f.write(f"  - Learning rate: {LEARNING_RATE}\n")
        f.write(f"  - Train samples: {len(train_paths)}\n")
        f.write(f"  - Val samples: {len(val_paths)}\n\n")
        f.write(f"Final Metrics:\n")
        f.write(f"  - Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"  - Precision: {metrics['precision']:.4f}\n")
        f.write(f"  - Recall:    {metrics['recall']:.4f}\n")
        f.write(f"  - F1-Score:  {metrics['f1_score']:.4f}\n")
        f.write(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}\n")
    
    print(f"✓ Saved metrics to {metrics_file}")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nModel saved to: {MODELS_DIR}/best_model_stage1.pth")
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"\nGenerated plots:")
    print(f"  - training_history.png")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - metrics_summary.png")
    print(f"  - metrics.txt")

if __name__ == '__main__':
    main()