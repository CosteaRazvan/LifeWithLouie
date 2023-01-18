from Parameters import *
import cv2 as cv
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torchvision.io import read_image

class Preprocess():
    def __init__(self, params:Parameters):
        self.params = params

    def crop_examples(self):
        n_pos = 0
        n_neg = 0
        num_negative_per_image = self.params.number_negative_examples // self.params.number_positive_examples

        if self.params.use_flip_images: num_negative_per_image +=1
        
        for char in ['andy', 'louie', 'ora', 'tommy']:
            print(f'Start {char} cropping')

            dir_path =  os.path.join(self.params.train_dir, f'{char}/')
            txt_path = os.path.join(self.params.train_dir, f'{char}_annotations.txt')

            f = open(txt_path, 'r')

            coords = {}
            for line in f:
                words = line.split(' ')

                if words[0] not in coords.keys():
                    coords[words[0]] = []

                if words[5] in [char, char+'\n']:
                    coords[words[0]].append(int(words[1]))
                    coords[words[0]].append(int(words[2]))
                    coords[words[0]].append(int(words[3]))
                    coords[words[0]].append(int(words[4]))

            f.close()

            # Char examples
            i = 0
            for (key, val) in coords.items():
                if len(val) > 0:
                    image = cv.imread(dir_path + key)

                    face = image[val[1]:val[3], val[0]:val[2]]
                    #face = cv.resize(face, (224, 224))

                    cv.imwrite(os.path.join(self.params.train_dir, f'task2_cropped/{char}/{i}.jpg'), face)
                    print(f'Saved {char} example number {i}')
                    i+=1
                    
                    if self.params.use_flip_images:
                        fliped_face = cv.flip(face, 1)
                        cv.imwrite(os.path.join(self.params.train_dir, f'task2_cropped/{char}/{i}.jpg'), fliped_face)
                        print(f'Saved {char} example number {i}')
                        i+=1
                        

            # Positive and negative examples
            f = open(txt_path, 'r')

            for line in f:
                words = line.split(' ')

                image = cv.imread(dir_path + words[0])

                val = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]

                face = image[val[1]:val[3], val[0]:val[2]]
                face = cv.resize(face, (self.params.dim_window, self.params.dim_window))

                cv.imwrite(os.path.join(self.params.dir_pos_examples, f'{n_pos}.jpg'), face)
                print(f'Saved positive example number {n_pos}')
                n_pos+=1

                if self.params.use_flip_images:
                        fliped_face = cv.flip(face, 1)
                        cv.imwrite(os.path.join(self.params.dir_pos_examples, f'{n_pos}.jpg'), fliped_face)
                        print(f'Saved positive example number {n_pos}')
                        n_pos+=1

                num_rows = image.shape[0]
                num_cols = image.shape[1]

                X, Y = [], []
                x_high = num_cols - self.params.dim_window
                y_high = num_rows - self.params.dim_window

                for k in range(num_negative_per_image):
                    x = np.random.randint(low=0, high=x_high)
                    while x >= val[0] and x <= val[2]:
                        x = np.random.randint(low=0, high=x_high)

                    y = np.random.randint(low=0, high=y_high)
                    while y >= val[1] and y <= val[3]:
                        y = np.random.randint(low=0, high=y_high)
                    
                    X.append(x)
                    Y.append(y)
                
                for idx in range(len(Y)):
                    patch = image[Y[idx]: Y[idx] + self.params.dim_window, X[idx]: X[idx] + self.params.dim_window]

                    cv.imwrite(os.path.join(self.params.dir_neg_examples, f'{n_neg}.jpg'), patch)
                    print(f'Saved negative example number {n_neg}')
                    n_neg+=1

            f.close()

        print(f'Number of positive examples: {n_pos}')
        print(f'Number of negative examples: {n_neg}')

    def crop_valid_data(self):
        txt_path = os.path.join(self.params.valid_dir, 'task1_gt_validare.txt')
        
        f = open(txt_path, 'r')
        n_val_pos = 0
        n_val_neg = 0
        for line in f:
            words = line.split(' ')
            image = cv.imread(os.path.join(self.params.dir_valid_images, words[0]))
            val = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]
            face = image[val[1]:val[3], val[0]:val[2]]
            face = cv.resize(face, (self.params.dim_window, self.params.dim_window))

            cv.imwrite(os.path.join(self.params.valid_dir, f'data_crop/positive/{n_val_pos}.jpg'), face)
            print(f'Val pos image no {n_val_pos}')
            n_val_pos+=1

            num_rows = image.shape[0]
            num_cols = image.shape[1]

            X, Y = [], []
            x_high = num_cols - self.params.dim_window
            y_high = num_rows - self.params.dim_window

            for k in range(2):
                x = np.random.randint(low=0, high=x_high)
                while x >= val[0] and x <= val[2]:
                    x = np.random.randint(low=0, high=x_high)

                y = np.random.randint(low=0, high=y_high)
                while y >= val[1] and y <= val[3]:
                    y = np.random.randint(low=0, high=y_high)
                
                X.append(x)
                Y.append(y)
            
            for idx in range(len(Y)):
                patch = image[Y[idx]: Y[idx] + self.params.dim_window, X[idx]: X[idx] + self.params.dim_window]
                cv.imwrite(os.path.join(self.params.valid_dir, f'data_crop/negative/{n_val_neg}.jpg'), patch)
                print(f'Val neg image no {n_val_neg}')
                n_val_neg+=1

        print(f'Total valid images: {n_val_neg+n_val_pos}')

    def crop_valid_data_task2(self):
        for char in ['andy', 'louie', 'ora', 'tommy']:
            txt_path = os.path.join(self.params.valid_dir, f'task2_{char}_gt_validare.txt')
        
            f = open(txt_path, 'r')
            i=0
            for line in f:
                words = line.split(' ')
                image = cv.imread(os.path.join(self.params.dir_valid_images, words[0]))
                val = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]
                face = image[val[1]:val[3], val[0]:val[2]]
                face = cv.resize(face, (224, 224))
                cv.imwrite(os.path.join(self.params.valid_dir, f'task2_cropped/{char}/{i}.jpg'), face)
                print(f'Valid image {i}')
                i+=1

    def get_data_loaders(self, batch_size=32):
        transformations = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        training_data = datasets.ImageFolder(os.path.join(self.params.train_dir, 'task2_cropped'), transformations)
        validation_data = datasets.ImageFolder(os.path.join(self.params.valid_dir, 'task2_cropped'), transformations)

        training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)

        return training_loader, validation_loader

       
