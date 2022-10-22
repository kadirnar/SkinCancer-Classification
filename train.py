import torch
import torch.nn as nn
import torch.optim as optim
import time
from data.dataloader import SkinCancerDataset
from backbone.model import SkinCancerResnet18

class Trainer:
    def __init__(self, num_epochs, num_classes):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.start = time.time()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.data_loader()
        self.get_model()
        self.hyperparameter_parameters()
        
    def data_loader(self):
        self.train_data_loader = SkinCancerDataset(64, "dataset").get_train_loader()
        self.test_data_loader = SkinCancerDataset(16, "dataset").get_test_loader()

    def get_model(self):
        self.model = SkinCancerResnet18(2).get_model()
        self.best_model_wts = self.model.state_dict()

    def hyperparameter_parameters(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001)

    def train_model(self):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            print('Epoch {}/{}'.format(epoch, self.num_epochs - 1))
            print('-' * 10)
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in self.train_data_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / len(self.train_data_loader.dataset)
                epoch_acc = running_corrects.double() / len(self.train_data_loader.dataset)
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = self.model.state_dict()


        time_elapsed = time.time() - self.start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(self.best_acc))
        self.model.load_state_dict(self.best_model_wts)
        return self.model
    
if __name__ == "__main__":
    trainer = Trainer(num_classes=2, num_epochs=10)
    trainer.train_model()