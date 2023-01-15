from tqdm import tqdm
from Network import ResNet, Block
from Parameters import *
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob
import cv2 as cv
from copy import deepcopy
import timeit
from PreprocessingData import Preprocess
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image as im

class Recognizer:
    def __init__(self, params: Parameters):
        self.params = params
        self.best_model = None
        self.softmax = nn.Softmax(dim=1)

    def train_step(self, model, training_loader, optimizer, criterion, device):
        model.train()
        print('Training')
        train_step_loss = 0.0
        train_step_correct = 0
        n = 0
        for i, data in tqdm(enumerate(training_loader), total=len(training_loader)):
            n += 1
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass.
            outputs = model(image)

            # Calculate the loss.
            loss = criterion(outputs, labels)
            train_step_loss += loss.item()

            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            train_step_correct += (preds == labels).sum().item()

            # Backpropagation
            loss.backward()

            # Update the weights.
            optimizer.step()
    
        # Loss and accuracy
        epoch_loss = train_step_loss / n
        epoch_acc = 100. * (train_step_correct / len(training_loader.dataset))

        return epoch_loss, epoch_acc

    def valid_step(self, model, valid_loader, criterion, device):
        model.eval()
        print('Validation')

        valid_step_loss = 0.0
        valid_step_correct = 0
        n = 0
        with torch.no_grad():
            for i, data in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                n += 1
                
                image, labels = data
                image = image.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(image)

                # Loss
                loss = criterion(outputs, labels)
                valid_step_loss += loss.item()

                # Accuracy
                _, preds = torch.max(outputs.data, 1)
                valid_step_correct += (preds == labels).sum().item()
            
        # Loss and accuracy
        epoch_loss = valid_step_loss / n
        epoch_acc = 100. * (valid_step_correct / len(valid_loader.dataset))
        return epoch_loss, epoch_acc

    def train(self, training_loader, validation_loader, batch_size):
        epochs = 70
        learning_rate = 0.001
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        resnet_file_name = os.path.join(self.params.saved_dir, f'task2/best_model.pth')

        if os.path.exists(resnet_file_name):
            self.best_model = ResNet(3, Block, 18, 4)
            self.best_model.load_state_dict(torch.load(resnet_file_name)['model_state_dict'])
            self.best_model.eval()
            print('Best net loaded')
            return

        print(f'Start training on {device}')
        print('Hyperparameters:')
        print(f'epochs: {epochs}')
        print(f'batch_size: {batch_size}')
        print(f'learning_rate: {learning_rate}')

        model = ResNet(image_channels=3, num_layers=18, block=Block, num_classes=4).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []

        best_valid_loss = float('inf')

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')

            train_epoch_loss, train_epoch_acc = self.train_step(model, training_loader, optimizer, criterion, device)
            valid_epoch_loss, valid_epoch_acc = self.valid_step(model, validation_loader, criterion, device)

            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)
            train_acc.append(train_epoch_acc)
            valid_acc.append(valid_epoch_acc)

            print(f"Training loss = {train_epoch_loss:.3f}     |   Training acc = {train_epoch_acc:.3f}")
            print(f"Validation loss = {valid_epoch_loss:.3f}   |   Validation acc = {valid_epoch_acc:.3f}")

            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                print(f'New best valid loss: {best_valid_loss:.3f}')
                print(f'Saving the model...')
                torch.save({'model_state_dict': model.state_dict()}, 
                    os.path.join(self.params.saved_dir, f'task2/best_model.pth'))

            print('-'*50)
            
        # Accuracy plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_acc, color='blue', label='train accuracy')
        plt.plot(valid_acc, color='orange', label='valid accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(os.path.join(self.params.saved_dir, 'task2/accuracy.eps'))
        
        # Loss plots.
        plt.figure(figsize=(10, 7))
        plt.plot(train_loss, color='blue', label='train loss')
        plt.plot(valid_loss, color='orange', label='validloss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(self.params.saved_dir, 'task2/loss.eps'))

        print('Done training')
        

    def run(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        detections_chars = {
            'andy': [],
            'louie': [],
            'ora': [],
            'tommy': []
        }

        scores_chars = {
            'andy': [],
            'louie': [],
            'ora': [],
            'tommy': []
        }

        file_names_chars = {
            'andy': [],
            'louie': [],
            'ora': [],
            'tommy': []
        }

        detections = np.load(os.path.join(self.params.sol_dir, 'task1/detections_all_faces.npy'))
        scores = np.load(os.path.join(self.params.sol_dir, 'task1/scores_all_faces.npy'))
        file_names = np.load(os.path.join(self.params.sol_dir, 'task1/file_names_all_faces.npy'))

        chars = ['andy', 'louie', 'ora', 'tommy']
        num_detections = len(detections)

        print('Start recognition process')
        for i in range(num_detections):
            image = cv.imread(os.path.join(self.params.dir_valid_images, file_names[i]))
            x_min, y_min = detections[i][0], detections[i][1]
            x_max, y_max = detections[i][2], detections[i][3]
            detection = image[y_min: y_max, x_min: x_max]
            detection = im.fromarray(detection)

            transformations = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            detection = transformations(detection).float().unsqueeze_(0)
            detection.to(device)

            with torch.no_grad():
                output = self.best_model(detection)
            
            output = output.to(device)
            output_prob = self.softmax(output)
            
            prob_max, prob_max_idx = torch.max(output_prob, 1)

            if prob_max > self.params.recognition_threshold:
                detections_chars[chars[prob_max_idx]].append(detections[i])
                scores_chars[chars[prob_max_idx]].append(scores[i])
                file_names_chars[chars[prob_max_idx]].append(file_names[i])

            print(f'Process detection {i}/{num_detections}')

        print('Saving solutions')
        for char in chars:
            np.save(os.path.join(self.params.sol_dir, f'task2/detections_{char}.npy'), detections_chars[char])
            np.save(os.path.join(self.params.sol_dir, f'task2/scores_{char}.npy'), scores_chars[char])
            np.save(os.path.join(self.params.sol_dir, f'task2/file_names_{char}.npy'), file_names_chars[char])

        print('End recognition process')

    def run_model(self):
        resnet_file_name = os.path.join(self.params.saved_dir, f'task2/best_model.pth')

        if os.path.exists(resnet_file_name):
            self.best_model = ResNet(3, Block, 18, 4)
            self.best_model.load_state_dict(torch.load(resnet_file_name)['model_state_dict'])
            self.best_model.eval()
            print('Best net loaded')

        


            

            
        
