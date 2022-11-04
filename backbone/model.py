import torchvision.models as models
from torch import nn
import torch


class SkinCancerResnet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(pretrained=False).to(self.device)
    
    def model_parameters(self):
        for param in self.model.parameters():
            param.requires_grad = False
        return self.model.fc
    
    def model_sequence(self):
        return nn.Sequential(
            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),  
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
            nn.LogSoftmax(dim=1)
        )
    
    def get_model(self):
        self.model.fc = self.model_sequence()
        return self.model.to(self.device)
    