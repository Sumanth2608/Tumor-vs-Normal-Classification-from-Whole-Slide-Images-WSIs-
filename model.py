import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class TumorClassifier(nn.Module):
    """
    ResNet-18 based model for tumor vs normal classification.
    Uses pretrained ResNet-18 with modified final layer for binary classification.
    """
    def __init__(self, num_classes=2, pretrained=True, freeze_backbone=False):
        super(TumorClassifier, self).__init__()
        # Load pretrained ResNet-18
        self.resnet = models.resnet18(pretrained=pretrained)

        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

        # Add dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Pass through ResNet backbone
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)

        # Apply dropout before final classification
        x = self.dropout(x)
        x = self.resnet.fc(x)
        return x

def get_model(num_classes=2, pretrained=True, freeze_backbone=False):
    """
    Factory function to get the ResNet-18 based model.

    Args:
        num_classes (int): Number of output classes (default: 2 for binary classification)
        pretrained (bool): Whether to use pretrained weights (default: True)
        freeze_backbone (bool): Whether to freeze ResNet backbone layers (default: False)

    Returns:
        TumorClassifier: The model instance
    """
    return TumorClassifier(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)