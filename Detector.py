from Parameters import *
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import glob
import cv2 as cv
import pickle
from copy import deepcopy
import timeit
from skimage.feature import hog

from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import shuffle
from PreprocessingData import Preprocess

class Detector:
    def __init__(self, params:Parameters):
        self.params = params
        self.best_model = None
        self.preprocessing: Preprocess = Preprocess(params)

    def get_positive_features(self):
        images_path = os.path.join(self.params.dir_pos_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process positive example number {i}')

            img = cv.imread(files[i])
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            features = hog(img_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            
            positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_positive_val_features(self):
        images_path = os.path.join(self.params.valid_dir, 'data_crop/positive/*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        positive_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process positive example number {i}')

            img = cv.imread(files[i])
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            features = hog(img_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)
            
            positive_descriptors.append(features)

        positive_descriptors = np.array(positive_descriptors)
        return positive_descriptors

    def get_negative_features(self):
        images_path = os.path.join(self.params.dir_neg_examples, '*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process negative example number {i}')

            img = cv.imread(files[i])
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            features = hog(img_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            negative_descriptors.append(features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def get_negative_val_features(self):
        images_path = os.path.join(self.params.valid_dir, 'data_crop/negative/*.jpg')
        files = glob.glob(images_path)
        num_images = len(files)
        negative_descriptors = []

        print(f'Compute positive descriptors for {num_images} images')
        for i in range(num_images):
            print(f'Process negative example number {i}')

            img = cv.imread(files[i])
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            features = hog(img_gray, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                           cells_per_block=(2, 2), feature_vector=True)

            negative_descriptors.append(features)

        negative_descriptors = np.array(negative_descriptors)
        return negative_descriptors

    def train_classifier(self, X, y):
        svm_file_name = os.path.join(self.params.saved_dir, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))
        if os.path.exists(svm_file_name):
            self.best_model = pickle.load(open(svm_file_name, 'rb'))
            return

        X, y = shuffle(X, y)

        best_accuracy = 0
        best_c = 0
        best_model = None
        Cs = [10 ** -5, 10 ** -4,  10 ** -3,  10 ** -2, 10 ** -1, 10 ** 0]
        for c in Cs:
            print(f'Train a classifier for c = {c}')

            model = LinearSVC(C=c)
            model.fit(X, y)

            acc = model.score(X, y)
            print(acc)

            if acc > best_accuracy:
                best_accuracy = acc
                best_c = c
                best_model = deepcopy(model)

        acc = best_model.score(X, y)
        print(f'Final model acc = {acc}')

        print('Performanta clasificatorului optim pt c = %f' % best_c)
        # salveaza clasificatorul
        pickle.dump(best_model, open(svm_file_name, 'wb'))

        # vizualizeaza cat de bine sunt separate exemplele pozitive de cele negative dupa antrenare
        # ideal ar fi ca exemplele pozitive sa primeasca scoruri > 0, iar exemplele negative sa primeasca scoruri < 0
        scores = best_model.decision_function(X)
        self.best_model = best_model
        positive_scores = scores[y > 0]
        negative_scores = scores[y <= 0]


        plt.plot(np.sort(positive_scores))
        plt.plot(np.zeros(len(positive_scores)))
        plt.plot(np.sort(negative_scores))
        plt.xlabel('Nr example antrenare')
        plt.ylabel('Scor clasificator')
        plt.title('Distributia scorurilor clasificatorului pe exemplele de antrenare')
        plt.legend(['Scoruri exemple pozitive', '0', 'Scoruri exemple negative'])
        plt.savefig(os.path.join(self.params.saved_dir, 'distr_train.eps'))
        plt.show()

    def run_svm(self, X, y):
        svm_file_name = os.path.join(self.params.saved_dir, 'best_model_%d_%d_%d' %
                                     (self.params.dim_hog_cell, self.params.number_negative_examples,
                                      self.params.number_positive_examples))

        if os.path.exists(svm_file_name) == False:
            return

        model = pickle.load(open(svm_file_name, 'rb'))

        y_hat = model.predict(X)
        cr = classification_report(y_true=y, y_pred=y_hat)
        print(cr)

        w = model.coef_.T
        bias = model.intercept_[0]
        y_hat = []

        for x in X:
            score = np.dot(x, w)[0] + bias
            if score > 0:
                y_hat.append(1)
            else:
                y_hat.append(0)

        acc = accuracy_score(y, y_hat)
        print(acc)

        
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
    
    def run(self):
        test_images_path = os.path.join(self.params.dir_valid_images, '*.jpg')
        test_files = sorted(glob.glob(test_images_path))
        detections = None  
        scores = np.array([])  
        file_names = np.array([])

        w = self.best_model.coef_.T
        bias = self.best_model.intercept_[0]
        num_test_images = len(test_files)

        for i in range(num_test_images):
            start_time = timeit.default_timer()
            print(f'Process test image {i}/{num_test_images}')
            img = cv.imread(test_files[i], cv.IMREAD_GRAYSCALE)

            image_scores = []
            image_detections = []

            ars = [0.60, 0.75, 0.90, 1, 1.15, 1.25, 1.40, 1.60, 1.70, 1.80, 1.95]
            scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 1.3, 1.6]

            for ar in ars:
                for scale in scales:

                    if ar < 1:
                        fx = ar * scale
                        fy = scale
                    else:
                        fx = scale
                        fy = (1/ar) * scale
                    
                    warped_img = cv.resize(img, dsize=None, fx=fx, fy=fy)

                    hog_descriptors = hog(warped_img, pixels_per_cell=(self.params.dim_hog_cell, self.params.dim_hog_cell),
                                        cells_per_block=(2, 2), feature_vector=False)

                    num_cols = warped_img.shape[1] // self.params.dim_hog_cell - 1
                    num_rows = warped_img.shape[0] // self.params.dim_hog_cell - 1
                    num_cell_in_template = self.params.dim_window // self.params.dim_hog_cell - 1

                    for y in range(0, num_rows - num_cell_in_template):
                        for x in range(0, num_cols - num_cell_in_template):
                            features = hog_descriptors[y:y + num_cell_in_template, x:x + num_cell_in_template].flatten()
                            score = np.dot(features, w)[0] + bias

                            if score > self.params.threshold:
                                x_min = int(x * self.params.dim_hog_cell)
                                y_min = int(y * self.params.dim_hog_cell)
                                x_max = int(x * self.params.dim_hog_cell + self.params.dim_window)
                                y_max = int(y * self.params.dim_hog_cell + self.params.dim_window)

                                x_min, y_min = int(x_min / fx), int(y_min / fy)
                                x_max, y_max = int(x_max / fx), int(y_max / fy)

                                image_detections.append([x_min, y_min, x_max, y_max])
                                image_scores.append(score)


            if len(image_scores) > 0:
                image_detections, image_scores = self.non_maximal_suppression(np.array(image_detections), np.array(image_scores), img.shape)

            if len(image_scores) > 0:
                if detections is None:
                    detections = image_detections
                else:
                    detections = np.concatenate((detections, image_detections))

                scores = np.append(scores, image_scores)
                short_name = os.path.basename(test_files[i])
                image_names = [short_name for ww in range(len(image_scores))]
                file_names = np.append(file_names, image_names)
            
            end_time = timeit.default_timer()
            print(f'Time for processing image {i}/{num_test_images} is {end_time-start_time} sec')

        return detections, scores, file_names

    def compute_average_precision(self, rec, prec):
        # functie adaptata din 2010 Pascal VOC development kit
        m_rec = np.concatenate(([0], rec, [1]))
        m_pre = np.concatenate(([0], prec, [0]))
        for i in range(len(m_pre) - 1, -1, 1):
            m_pre[i] = max(m_pre[i], m_pre[i + 1])
        m_rec = np.array(m_rec)
        i = np.where(m_rec[1:] != m_rec[:-1])[0] + 1
        average_precision = np.sum((m_rec[i] - m_rec[i - 1]) * m_pre[i])
        return average_precision

    def eval_detections(self, detections, scores, file_names):
        ground_truth_file = np.loadtxt(self.params.path_annotations_task_1, dtype='str')
        ground_truth_file_names = np.array(ground_truth_file[:, 0])
        ground_truth_detections = np.array(ground_truth_file[:, 1:], np.int)

        num_gt_detections = len(ground_truth_detections)  # numar total de adevarat pozitive
        gt_exists_detection = np.zeros(num_gt_detections)
        # sorteazam detectiile dupa scorul lor
        sorted_indices = np.argsort(scores)[::-1]
        file_names = file_names[sorted_indices]
        scores = scores[sorted_indices]
        detections = detections[sorted_indices]

        num_detections = len(detections)
        true_positive = np.zeros(num_detections)
        false_positive = np.zeros(num_detections)
        duplicated_detections = np.zeros(num_detections)

        for detection_idx in range(num_detections):
            indices_detections_on_image = np.where(ground_truth_file_names == file_names[detection_idx])[0]

            gt_detections_on_image = ground_truth_detections[indices_detections_on_image]
            bbox = detections[detection_idx]
            max_overlap = -1
            index_max_overlap_bbox = -1
            for gt_idx, gt_bbox in enumerate(gt_detections_on_image):
                overlap = self.intersection_over_union(bbox, gt_bbox)
                if overlap > max_overlap:
                    max_overlap = overlap
                    index_max_overlap_bbox = indices_detections_on_image[gt_idx]

            # clasifica o detectie ca fiind adevarat pozitiva / fals pozitiva
            if max_overlap >= 0.3:
                if gt_exists_detection[index_max_overlap_bbox] == 0:
                    true_positive[detection_idx] = 1
                    gt_exists_detection[index_max_overlap_bbox] = 1
                else:
                    false_positive[detection_idx] = 1
                    duplicated_detections[detection_idx] = 1
            else:
                false_positive[detection_idx] = 1

        cum_false_positive = np.cumsum(false_positive)
        cum_true_positive = np.cumsum(true_positive)

        rec = cum_true_positive / num_gt_detections
        prec = cum_true_positive / (cum_true_positive + cum_false_positive)
        average_precision = self.compute_average_precision(rec, prec)
        plt.plot(rec, prec, '-')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Average precision %.3f' % average_precision)
        plt.savefig(os.path.join(self.params.saved_dir, 'precizie_medie.eps'))
        plt.show()