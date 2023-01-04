from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pdb
import pickle
import ntpath
from copy import deepcopy
import timeit
from skimage.feature import hog

class Detector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None

    def get_positive_features(self):
        # in aceasta functie calculam descriptorii pozitivi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor pozitive
        # iar D - dimensiunea descriptorului
        # D = (params.dim_window/params.dim_hog_cell - 1) ^ 2 * params.dim_descriptor_cell (fetele sunt patrate)

        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process positive example number {i}')

            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_features(self):
        # in aceasta functie calculam descriptorii negativi
        # vom returna un numpy array de dimensiuni NXD
        # unde N - numar exemplelor negative
        # iar D - dimensiunea descriptorului
        # avem 274 de imagini negative, vream sa avem self.params.number_negative_examples (setat implicit cu 10000)
        # de exemple negative, din fiecare imagine vom genera aleator self.params.number_negative_examples // 274
        # patch-uri de dimensiune 36x36 pe care le vom considera exemple negative

        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process negative example number {i}')

            img = cv.imread(files[i], cv.IMREAD_GRAYSCALE)
            features = hog(img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            negative_descriptors.append(features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, training_examples, train_labels):
        svm_file_name = os.path.join(self.params.dir_save_files, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print(f'Train a classifier for c = {c}')

            model = LinearSVC(C=c)
            model.fit(training_examples, train_labels)

            acc = model.score(training_examples, train_labels)
            print(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(training_examples)
        self.best_model = best_model
        positive_scores = scores[train_labels > 0]
        negative_scores = scores[train_labels <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.show()

    def intersection_over_union(self, bbox_a, bbox_b):
        x_a = max(bbox_a[0], bbox_b[0])
        y_a = max(bbox_a[1], bbox_b[1])
        x_b = min(bbox_a[2], bbox_b[2])
        y_b = min(bbox_a[3], bbox_b[3])

        inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

        box_a_area = (bbox_a[2] - bbox_a[0] + 1) * (bbox_a[3] - bbox_a[1] + 1)
        box_b_area = (bbox_b[2] - bbox_b[0] + 1) * (bbox_b[3] - bbox_b[1] + 1)

        iou = inter_area / float(box_a_area + box_b_area - inter_area)

        return iou

    def non_maximal_suppression(self, image_detections, image_scores, image_size):
        """
        Detectiile cu scor mare suprima detectiile ce se suprapun cu acestea dar au scor mai mic.
        Detectiile se pot suprapune partial, dar centrul unei detectii nu poate
        fi in interiorul celeilalte detectii.
        :param image_detections:  numpy array de dimensiune NX4, unde N este numarul de detectii.
        :param image_scores: numpy array de dimensiune N
        :param image_size: tuplu, dimensiunea imaginii
        :return: image_detections si image_scores care sunt maximale.
        """

        # xmin, ymin, xmax, ymax
        x_out_of_bounds = np.where(image_detections[:, 2] > image_size[1])[0]
        y_out_of_bounds = np.where(image_detections[:, 3] > image_size[0])[0]
        print(x_out_of_bounds, y_out_of_bounds)
        image_detections[x_out_of_bounds, 2] = image_size[1]
        image_detections[y_out_of_bounds, 3] = image_size[0]
        sorted_indices = np.flipud(np.argsort(image_scores))
        sorted_image_detections = image_detections[sorted_indices]
        sorted_scores = image_scores[sorted_indices]

        is_maximal = np.ones(len(image_detections)).astype(bool)
        iou_threshold = 0.3
        for i in range(len(sorted_image_detections) - 1):
            if is_maximal[i] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                for j in range(i + 1, len(sorted_image_detections)):
                    if is_maximal[j] == True:  # don't change to 'is True' because is a numpy True and is not a python True :)
                        if self.intersection_over_union(sorted_image_detections[i],sorted_image_detections[j]) > iou_threshold:is_maximal[j] = False
                        else:  # verificam daca centrul detectiei este in mijlocul detectiei cu scor mai mare
                            c_x = (sorted_image_detections[j][0] + sorted_image_detections[j][2]) / 2
                            c_y = (sorted_image_detections[j][1] + sorted_image_detections[j][3]) / 2
                            if sorted_image_detections[i][0] <= c_x <= sorted_image_detections[i][2] and \
                                    sorted_image_detections[i][1] <= c_y <= sorted_image_detections[i][3]:
                                is_maximal[j] = False
        return sorted_image_detections[is_maximal], sorted_scores[is_maximal]
