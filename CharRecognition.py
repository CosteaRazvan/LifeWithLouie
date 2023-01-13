from tqdm import tqdm
from Network import ResNet, Block
from Parameters import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import cv2 as cv
import pickle
from copy import deepcopy
import timeit
from PreprocessingData import Preprocess
import torch.nn as nn
import torch

class CharRecognition:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None

    def train_step(self, model, training_loader, optimizer, criterion, device):
        model.train()
        print('Training')
        train_running_loss = 0.0
        train_running_correct = 0
        counter = 0
        for i, data in tqdm(enumerate(training_loader), total=len(training_loader)):
            counter += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            train_running_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_running_correct += (preds == labels).sum().item()

            # Backpropagation
            loss.backward()

            # Update the weights.
            optimizer.step()
    
        # Loss and accuracy for the complete epoch.
        epoch_loss = train_running_loss / counter
        epoch_acc = 100. * (train_running_correct / len(training_loader.dataset))

        return epoch_loss, epoch_acc

    def validate(self, model, valid_loader, criterion, device):
        model.eval()
        print('Validation')

        valid_running_loss = 0.0
        valid_running_correct = 0
        counter = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                counter += 1
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)
                # Forward pass.
                outputs = model(image)
                # Calculate the loss.
                loss = criterion(outputs, labels)
                valid_running_loss += loss.item()
                # Calculate the accuracy.
                _, preds = torch.max(outputs.data, 1)
                valid_running_correct += (preds == labels).sum().item()
            
        # Loss and accuracy for the complete epoch.
        epoch_loss = valid_running_loss / counter
        epoch_acc = 100. * (valid_running_correct / len(valid_loader.dataset))
        return epoch_loss, epoch_acc

    def train(self, training_loader, validation_loader):
        epochs = 20
        batch_size = 32
        learning_rate = 0.01
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        resnet_file_name = os.path.join(self.params.saved_dir, f'best_model_{epochs}_{batch_size}.pth')

        if os.path.exists(resnet_file_name):
            self.best_model = torch.load(resnet_file_name)
            return

        model = ResNet(image_channels=3, num_layers=18, block=Block, num_classes=4).to(device)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        best_valid_loss = float('inf')

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')

            train_epoch_loss, train_epoch_acc = self.train_step(model, training_loader, optimizer, criterion, device)
            valid_epoch_loss, valid_epoch_acc = self.validate(model, validation_loader, criterion,device)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
            print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                print(f'New best valid loss: {best_valid_loss:.3f}')
                print(f'Saving the model...')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': criterion
                    }, os.path.join(self.params.saved_dir, f'task2/best_model_{epochs}_{batch_size}.pth'))

            print('-'*50)
            
            
        # Save the loss and accuracy plots.
        self.save_plots(train_acc, valid_acc, train_loss, valid_loss)
        print('TRAINING COMPLETE')

        

    def save_plots(self, train_acc, valid_acc, train_loss, valid_loss, name=None):
        # Accuracy plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_acc, color='tab:blue', linestyle='-', label='train accuracy')
        plt.plot(valid_acc, color='tab:red', linestyle='-', label='valid accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.params.saved_dir, 'task2/accuracy.eps'))
        
        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='tab:blue', linestyle='-', label='train loss')
        plt.plot(valid_loss, color='tab:red', linestyle='-', label='validloss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.params.saved_dir, 'task2/loss.eps'))

        
