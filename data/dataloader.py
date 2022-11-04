import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils.data_utils import read_yaml

class SkinCancerDataset:
    def __init__(self, batch_size, data_dir):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
        }
        self.image_datasets = {
            "train": datasets.ImageFolder(self.data_dir + "/train", transform=self.data_transforms["train"]),
            "test": datasets.ImageFolder(self.data_dir + "/test", transform=self.data_transforms["test"]),
            "val": datasets.ImageFolder(self.data_dir + "/val", transform=self.data_transforms["val"])
        }
        
        self.data_loaders = {
            'train': torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=self.batch_size, shuffle=True),
            'test': torch.utils.data.DataLoader(self.image_datasets['test'], batch_size=self.batch_size, shuffle=True),
            'val': torch.utils.data.DataLoader(self.image_datasets['val'], batch_size=self.batch_size, shuffle=True)
        }
        
    def get_train_loader(self):
        return self.data_loaders['train']

    def get_test_loader(self):
        return self.data_loaders['test']

    def get_val_loader(self):
        return self.data_loaders['val']


class SkinCancerYaml:
    def __init__(self, yaml_file):
        self.yaml_file = yaml_file
        
    def get_yaml(self):
        return read_yaml(self.yaml_file)
    