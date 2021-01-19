from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

class Model:
    """
    Main class where training, validation, and prediction is done in separate functions.

    """
    def __init__(self, model, criterion: nn.CrossEntropyLoss, optimizer: torch.optim, train_loader, validation_loader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_on_gpu = bool(torch.cuda.is_available())
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.train_losses = []
        self.validation_losses = []
        if self.train_on_gpu:
            self.model.cuda()
            print('CUDA is available! Training on GPU')
        else:
            print('Train on CPU because CUDA isn\'t available.')
    
    def train(self) -> float:
        train_loss = 0.0

        self.model.train()
        for data, target in self.train_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()*data.size(0)

        train_loss = train_loss/len(self.train_loader.sampler)
        return train_loss
    
    def validate(self) -> float:
        validation_loss = 0.0
        
        self.model.eval()
        
        for data, target in self.validation_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()

            output = self.model(data)
            loss = self.criterion(output, target)
            validation_loss += loss.item()*data.size(0)
            
        validation_loss = validation_loss / len(self.validation_loader.sampler)

        return validation_loss

    def predict(self, test_loader, path_to_weights: str) -> Tuple[float, pd.DataFrame]:
        self.load_model_weights(path_to_weights)
        self.model.eval()

        test_labels = test_loader.labels['Filename'].values.tolist()
        predictions = []
        targets = []
        correct = 0
        total = 0

        for data, target in test_loader:
            if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()

            output = self.model(data)

            _, predicted = torch.max(output.data, 1)
            targ = target[:].cpu().numpy()
            targets = np.concatenate([targets, targ])

            pred = predicted[:].cpu().numpy()
            predictions = np.concatenate([predictions, pred])

            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        test_acc = round((100 * correct / total), 5)
        print('Test Accuracy: {} %'.format(test_acc))

        return test_acc, pd.DataFrame(list(zip(test_labels, predictions)), columns=['Filename','Prediction'])
    
    def load_model_weights(self, path_to_weights: str):
        self.model.load_state_dict(torch.load(path_to_weights))