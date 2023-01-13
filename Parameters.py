import os

class Parameters:
    def __init__(self):
        self.train_dir = './antrenare/'
        self.valid_dir = './validare/'
        self.saved_dir = './saved/'
        self.test_dir = './test/'
        self.sol_dir = './solution_files/'

        self.dir_valid_images = os.path.join(self.valid_dir, 'Validare/')
        self.dir_pos_examples = os.path.join(self.train_dir, 'exemple_pozitive')
        self.dir_neg_examples = os.path.join(self.train_dir, 'exemple_negative')

        self.path_annotations_task_1 = os.path.join(self.valid_dir, 'task1_gt_validare.txt')
        
        self.dim_window = 60
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.overlap = 0.3
        self.number_positive_examples = 3994  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.has_annotations = False
        self.threshold = 0
        self.use_flip_images = True