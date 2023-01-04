from Parameters import *
from PreprocessingData import *
from Detector import *

params: Parameters = Parameters()
params.dim_window_1_1 = [60, 60]
params.dim_window_1_2 = [60, 120]  
params.dim_window = 60 
params.dim_hog_cell = 6  # 12
params.overlap = 0.3
params.number_positive_examples = 3994  # numarul exemplelor pozitive
params.number_negative_examples = 8000  # numarul exemplelor negative

params.threshold = 4.5 # toate ferestrele cu scorul > threshold si maxime locale devin detectii
params.has_annotations = True

params.use_hard_mining = False  # (optional)antrenare cu exemple puternic negative
params.use_flip_images = True  # adauga imaginile cu fete oglindite

if params.use_flip_images:
    params.number_positive_examples *= 2
    params.number_negative_examples *= 2

preprocess: Preprocess = Preprocess(params)
#TODO: crop if needed
#preprocess.crop_examples()

detector: Detector = Detector(params)

# positive features
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

# negative features
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
