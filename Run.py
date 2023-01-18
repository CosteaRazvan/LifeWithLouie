from Parameters import *
from PreprocessingData import *
from Detector import *
from Recognizer import *

params: Parameters = Parameters() 
params.dim_window = 60 
params.dim_hog_cell = 6  # 12
params.overlap = 0.3
params.number_positive_examples = 7236  # numarul exemplelor pozitive
params.number_negative_examples = 14472  # numarul exemplelor negative

params.threshold = 3
params.has_annotations = True

params.use_flip_images = True  

if params.use_flip_images:
    params.number_positive_examples *= 2

# Crop examples for both tasks
preprocess: Preprocess = Preprocess(params)
# uncomment if you want to crop the images for training
# preprocess.crop_examples()

############################Task1####################################

detector: Detector = Detector(params)

# Extract features from examples

positive_features_path = os.path.join(params.saved_dir, 'positive_example_descr_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_positive_examples) + '.npy')

if os.path.exists(positive_features_path):
    positive_features = np.load(positive_features_path)
    print('Loaded positive features')
else:
    print('Get positive features:')
    positive_features = detector.get_positive_features()
    np.save(positive_features_path, positive_features)
    print(f'Positive features are saved in {positive_features_path}')

negative_features_path = os.path.join(params.saved_dir, 'negative_example_descr_' + str(params.dim_hog_cell) + '_' +
                        str(params.number_negative_examples) + '.npy')

if os.path.exists(negative_features_path):
    negative_features = np.load(negative_features_path)
    print('Loaded negative features')
else:
    print('Get negative features:')
    negative_features = detector.get_negative_features()
    np.save(negative_features_path, negative_features)
    print(f'negative features are saved in {negative_features_path}')

# Fit the linear classifier
train_data = np.concatenate((np.squeeze(positive_features), np.squeeze(negative_features)), axis=0)
train_labels = np.concatenate((np.ones(params.number_positive_examples), np.zeros(negative_features.shape[0])))
detector.train_classifier(train_data, train_labels)

# Run on test data
test_images_path = 'path/to/test/image/dir/'
valid_images_path = os.path.join(params.dir_valid_images, '*.jpg')
detections, scores, file_names = detector.run(valid_images_path)

detection_path_task_1 = os.path.join(params.sol_dir, 'task1/detections_all_faces.npy')
scores_path_task_1 = os.path.join(params.sol_dir, 'task1/scores_all_faces.npy')
file_names_path_task_1 = os.path.join(params.sol_dir, 'task1/file_names_all_faces.npy')

np.save(detection_path_task_1, detections)
np.save(scores_path_task_1, scores)
np.save(file_names_path_task_1, file_names)

# ############################Task2####################################

recognizer: Recognizer = Recognizer(params)
batch_size = 32

# Load the data
train_loader, valid_loader = preprocess.get_data_loaders(batch_size=batch_size)

# Train the network
recognizer.train(train_loader, valid_loader, batch_size)

# Run on test data
recognizer.run()
