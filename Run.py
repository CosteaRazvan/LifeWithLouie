from Parameters import *
from PreprocessingData import *

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

preprocess.crop_examples()
