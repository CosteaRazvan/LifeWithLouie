from Parameters import *
import cv2 as cv
import numpy as np
import os
import glob


class Preprocess():
    def __init__(self, params:Parameters):
        self.params = params
    
    def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

    def resizeAndPad(img, size, padColor=0):
        h, w = img.shape[:2]
        sh, sw = size
        # interpolation method
        if h > sh or w > sw: # shrinking image
            interp = cv.INTER_AREA
        else: # stretching image
            interp = cv.INTER_CUBIC
        # aspect ratio of image
        aspect = w/h
        # compute scaling and pad sizing
        if aspect > 1: # horizontal image
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: # vertical image
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: # square image
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
        # set pad color
        if len(img.shape) == 3 and not isinstance(padColor, (list, tuple, 
            np.ndarray)): # color image but only one color provided
            padColor = [padColor]*3
            # scale and pad
            scaled_img = cv.resize(img, (new_w, new_h), interpolation=interp)
            scaled_img = cv.copyMakeBorder(scaled_img, pad_top, pad_bot, 
                pad_left, pad_right, borderType=cv.BORDER_CONSTANT,  
                value=padColor)
        return scaled_img







    def crop_examples(self):
        n_pos = 0
        n_neg = 0
        num_negative_per_image = self.params.number_negative_examples // self.params.number_positive_examples
        
        for char in ['andy', 'louie', 'ora', 'tommy']:
            print(f'Start {char} cropping')

            dir_path =  os.path.join(self.params.base_dir, f'{char}/')
            txt_path = os.path.join(self.params.base_dir, f'{char}_annotations.txt')

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

            # dim_window = self.params.dim_window_1_1
            # dir_pos_examples = self.params.dir_pos_examples_1_1
            # dir_neg_examples = self.params.dir_neg_examples_1_1
            # if char in ['andy', 'ora']:
            #     dim_window = self.params.dim_window_1_2
            #     dir_pos_examples = self.params.dir_pos_examples_1_2
            #     dir_neg_examples = self.params.dir_neg_examples_1_2

            # Char examples
            i = 0
            for (key, val) in coords.items():
                if len(val) > 0:
                    image = cv.imread(dir_path + key)

                    face = image[val[1]:val[3], val[0]:val[2]]
                    #face = cv.resize(face, (self.params.dim_window, self.params.dim_window))
                    face = self.image_resize(face, self.params.dim_window, self.params.dim_window)

                    cv.imwrite(os.path.join(self.params.base_dir, f'{char}_positive/{i}.jpg'), face)
                    #cv.imwrite(os.path.join(self.parmasdir_pos_examples, f'{n_pos}.jpg'), face)
                    i+=1
                    #n_pos+=1

                    if self.params.use_flip_images:
                        fliped_face = cv.flip(face, 1)
                        cv.imwrite(os.path.join(self.params.base_dir, f'{char}_positive/{i}.jpg'), fliped_face)
                        #cv.imwrite(os.path.join(dir_pos_examples, f'{n_pos}.jpg'), fliped_face)
                        i+=1
                        #n_pos+=1

            # Positive and negative examples
            f = open(txt_path, 'r')

            for line in f:
                words = line.split(' ')

                image = cv.imread(dir_path + words[0])

                val = [int(words[1]), int(words[2]), int(words[3]), int(words[4])]

                face = image[val[1]:val[3], val[0]:val[2]]
                #face = cv.resize(face, (self.params.dim_window, self.params.dim_window))
                face = self.image_resize(face, self.params.dim_window, self.params.dim_window)

                cv.imwrite(os.path.join(self.params.dir_pos_examples, f'{n_pos}.jpg'), face)
                n_pos+=1

                if self.params.use_flip_images:
                        fliped_face = cv.flip(face, 1)
                        cv.imwrite(os.path.join(self.params.dir_pos_examples, f'{n_pos}.jpg'), fliped_face)
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

                    if self.params.use_flip_images:
                        fliped_patch = cv.flip(patch, 1)
                        cv.imwrite(os.path.join(self.params.dir_neg_examples, f'{n_neg}.jpg'), fliped_patch)
                        print(f'Saved negative example number {n_neg}')
                        n_neg+=1

            f.close()

            
            # Negative examples
            # for (key, val) in coords.items():
            #     if len(val) > 0:
            #         image = cv.imread(dir_path + key)

            #         num_rows = image.shape[0]
            #         num_cols = image.shape[1]

            #         X, Y = [], []
            #         x_high = num_cols - dim_window[0]
            #         y_high = num_rows - dim_window[1]

            #         for k in range(num_negative_per_image):
            #             x = np.random.randint(low=0, high=x_high)
            #             while x >= val[0] and x <= val[2]:
            #                 x = np.random.randint(low=0, high=x_high)

            #             y = np.random.randint(low=0, high=y_high)
            #             while y >= val[1] and y <= val[3]:
            #                 y = np.random.randint(low=0, high=y_high)
                        
            #             X.append(x)
            #             Y.append(y)
                    
            #         for idx in range(len(Y)):
            #             patch = image[Y[idx]: Y[idx] + dim_window[1], X[idx]: X[idx] + dim_window[0]]

            #             cv.imwrite(os.path.join(dir_neg_examples, f'{n_neg}.jpg'), patch)
            #             print(f'Saved negative example number {n_neg}')
            #             n_neg+=1

            #             if self.params.use_flip_images:
            #                 fliped_patch = cv.flip(patch, 1)
            #                 cv.imwrite(os.path.join(dir_neg_examples, f'{n_neg}.jpg'), fliped_patch)
            #                 print(f'Saved negative example number {n_neg}')
            #                 n_neg+=1

        print(f'Number of positive examples: {n_pos}')
        print(f'Number of negative examples: {n_neg}')

