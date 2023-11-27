import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DenseNetBinaryClassifier(nn.Module):
    def __init__(self):
        super(DenseNetBinaryClassifier, self).__init__()
        # Load a pre-trained densenet
        self.densenet = models.densenet121(pretrained=True)
        
        # Replace classifier
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.densenet(x)

