import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class DenseNetBinaryClassifier(nn.Module):
    def __init__(self, logger=None):
        super(DenseNetBinaryClassifier, self).__init__()
        self.logger=logger
        self.densenet = models.densenet121(weights=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.densenet(x)

    def log_params(self):
        total_params = sum(p.numel() for p in self.densenet.parameters())
        try: 
            self.logger.info(f"Total parameters in the model: {total_params}")
        except:
            print(f"Total parameters in the model:{total_params}")