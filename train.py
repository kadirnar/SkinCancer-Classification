import torch
import torch.nn as nn
import torch.optim as optim
import time
from data.dataloader import SkinCancerDataset
from backbone.model import SkinCancerResnet18
import neptune.new as neptune
  
def init_neptune():
    run = neptune.init(project='common/pytorch-integration',
                       api_token='ANONYMOUS',
    )
    return run

run = init_neptune()
run["../dataset/train"].track_files("../dataset/train")
run["../dataset/test"].track_files("../dataset/test")
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
        self.train_data_loader = SkinCancerDataset(64, "../dataset").get_train_loader()
        self.test_data_loader = SkinCancerDataset(16, "../dataset").get_test_loader()
        self.val_data_loader = SkinCancerDataset(16, "../dataset").get_val_loader()

    def get_model(self):
        self.model = SkinCancerResnet18(self.num_classes).get_model()
        self.best_model_wts = self.model.state_dict()

    def hyperparameter_parameters(self):
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

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
                run["train/batch/loss"].log(epoch_loss)
                run["train/batch/acc"].log(epoch_acc) 
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    self.best_model_wts = self.model.state_dict()
                    self.val_model()

        time_elapsed = time.time() - self.start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test Acc: {:4f}'.format(best_acc))
        run["Summary"].log({"Best test Acc": best_acc})
        self.model.load_state_dict(self.best_model_wts)
        return self.model
    
    def val_model(self):
        self.model.eval()
        running_corrects = 0
        for inputs, labels in self.val_data_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        epoch_acc = running_corrects.double() / len(self.val_data_loader.dataset)
        print('Val Acc: {:.4f}'.format(epoch_acc))
        run["Val Acc"].log(epoch_acc)
        return self.model
    
if __name__ == "__main__":
    run["parameters"] = {
        "num_epochs": 25,
        "optimizer": "SGD",
        "loss_function": "NLLLoss",
        "Dropout": 0.3,
        "Conv2d": "5x5",
        
    }
    trainer = Trainer(25, 2)
    trainer.train_model()
    
    
    