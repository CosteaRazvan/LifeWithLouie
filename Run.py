from Parameters import *
from PreprocessingData import *
from Detector import *
from Visualize import *
from CharRecognition import *

params: Parameters = Parameters() 
params.dim_window = 60 
params.dim_hog_cell = 6  # 12
params.overlap = 0.3
params.number_positive_examples = 7236  # numarul exemplelor pozitive
params.number_negative_examples = 14472  # numarul exemplelor negative

params.threshold = 0
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2
    #params.number_negative_examples *= 2

# Crop examples
preprocess: Preprocess = Preprocess(params)
#TODO: crop if needed
#preprocess.crop_examples()
#preprocess.crop_valid_data_task2()

detector: Detector = Detector(params)

# Extract features from examples
# positive features
# positive_features_path = os.path.join(params.saved_dir, 'positive_example_descr_' + str(params.dim_hog_cell) + '_' +
#                         str(params.number_positive_examples) + '.npy')

# if os.path.exists(positive_features_path):
#     positive_features = np.load(positive_features_path)
#     print('Loaded positive features')
# else:
#     print('Get positive features:')
#     positive_features = detector.get_positive_features()
#     np.save(positive_features_path, positive_features)
#     print(f'Positive features are saved in {positive_features_path}')

# # negative features
# negative_features_path = os.path.join(params.saved_dir, 'negative_example_descr_' + str(params.dim_hog_cell) + '_' +
#                         str(params.number_negative_examples) + '.npy')

# if os.path.exists(negative_features_path):
#     negative_features = np.load(negative_features_path)
#     print('Loaded negative features')
# else:
#     print('Get negative features:')
#     negative_features = detector.get_negative_features()
#     np.save(negative_features_path, negative_features)
#     print(f'negative features are saved in {negative_features_path}')

# # Fit the linear classifier
# train_data = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
# train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
# detector.train_classifier(train_data, train_labels)

# Test on validation (fake test) data
# detections, scores, file_names = detector.run()

# detection_path_task_1 = os.path.join(params.sol_dir, 'task1/detections_all_faces.npy')
# scores_path_task_1 = os.path.join(params.sol_dir, 'task1/scores_all_faces.npy')
# file_names_path_task_1 = os.path.join(params.sol_dir, 'task1/file_names_all_faces.npy')

# np.save(detection_path_task_1, detections)
# np.save(scores_path_task_1, scores)
# np.save(file_names_path_task_1, file_names)

char_recognition: CharRecognition = CharRecognition(params) 
train_loader, valid_loader = preprocess.get_data_loaders(batch_size=32)
char_recognition.train(train_loader, valid_loader)

# Test svm on validation data

# #preprocess.crop_valid_data()
# positive_features_path_val = os.path.join(params.saved_dir, 'val_positive_example_descr_' + str(params.dim_hog_cell) + '_' +
#                         str(params.number_positive_examples) + '.npy')

# if os.path.exists(positive_features_path_val):
#     positive_features_val = np.load(positive_features_path_val)
#     print('Loaded positive val features')
# else:
#     print('Get positive features:')
#     positive_features_val = detector.get_positive_val_features()
#     np.save(positive_features_path_val, positive_features_val)
#     print(f'Positive features are saved in {positive_features_path_val}')

# # negative features
# negative_features_path_val = os.path.join(params.saved_dir, 'val_negative_example_descr_' + str(params.dim_hog_cell) + '_' +
#                         str(params.number_negative_examples) + '.npy')

# if os.path.exists(negative_features_path_val):
#     negative_features_val = np.load(negative_features_path_val)
#     print('Loaded negative val features')
# else:
#     print('Get negative features:')
#     negative_features_val = detector.get_negative_val_features()
#     np.save(negative_features_path_val, negative_features_val)
#     print(f'negative features are saved in {negative_features_path_val}') 

# X = np.concatenate((np.squeeze(positive_features_val), np.squeeze(negative_features_val)), axis=0)
# y = np.concatenate((np.ones(positive_features_val.shape[0]), np.zeros(negative_features_val.shape[0])))

# detector.run_svm(X, y)

# class Task2Dataset(Dataset):
#     def __init__(self, img_dir, label_dir, params:Parameters, transform=None, target_transform=None):
#         self.params = params
#         self.img_labels = pd.read_csv(os.path.join(self.params.train_dir, 'labels.csv'))
#         self.img_dir = os.path.join(self.params.train_dir, 'characters/')
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

# train_data = Task2Dataset()
