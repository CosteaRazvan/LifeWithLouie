import os

class Parameters:
    def __init__(self):
        self.base_dir = './antrenare/'
        self.dir_pos_examples_1_1 = os.path.join(self.base_dir, 'exemple_pozitive_1_1/')
        self.dir_pos_examples_1_2 = os.path.join(self.base_dir, 'exemple_pozitive_1_2/')
        self.dir_neg_examples_1_1 = os.path.join(self.base_dir, 'exemple_negative_1_1/')
        self.dir_neg_examples_1_2 = os.path.join(self.base_dir, 'exemple_negative_1_2/')

        self.dir_pos_examples = os.path.join(self.base_dir, 'exemple_pozitive')
        self.dir_neg_examples = os.path.join(self.base_dir, 'exemple_negative')

        self.dir_test_examples = os.path.join(self.base_dir,'exempleTest/CMU+MIT')  # 'exempleTest/CursVA'   'exempleTest/CMU+MIT'
        self.path_annotations = os.path.join(self.base_dir, 'exempleTest/CMU+MIT_adnotari/ground_truth_bboxes.txt')
        self.dir_save_files = os.path.join(self.base_dir, 'salveazaFisiere')
        if not os.path.exists(self.dir_save_files):
            os.makedirs(self.dir_save_files)
            print('directory created: {} '.format(self.dir_save_files))
        else:
            print('directory {} exists '.format(self.dir_save_files))

        # self.dim_window_1_1 = [36, 36]
        # self.dim_window_1_2 = [36, 72]  
        self.dim_window = 36
        self.dim_hog_cell = 6  # dimensiunea celulei
        self.dim_descriptor_cell = 36  # dimensiunea descriptorului unei celule
        self.overlap = 0.3
        self.number_positive_examples = 3994  # numarul exemplelor pozitive
        self.number_negative_examples = 10000  # numarul exemplelor negative
        self.overlap = 0.3
        self.has_annotations = False
        self.threshold = 0
        self.use_flip_images = True