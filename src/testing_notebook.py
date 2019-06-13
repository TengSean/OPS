# %%
# *Function library import
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from sklearn.model_selection import train_test_split

from tqdm import tqdm_notebook as tqdm
import os

# import warning
import datetime
import time

import cv2
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
pd.options.display.max_columns = 999


IMG_DIR = "../input/iwildcam-2019-fgvc6/"
RESIZE_IMG_DIR = "../input/reducing-image-sizes-to-32x32/"
WEIGHTS_DIR = "../input/densenet-keras/"
# print(os.listdir('../input'))
# %%

print("csv: {}".format(os.listdir(RAW_INPUT_DIR)))
print("Resize: {}".format(os.listdir(RESIZE_INPUT_DIR)))
print("Model: {}".format(os.listdir((MODEL_WEIGHTS_DIR))))
# %%
# *Data loading 
Train_df = pd.read_csv(RAW_INPUT_DIR + '/train.csv')
Test_df = pd.read_csv(RAW_INPUT_DIR + '/test.csv')

print("Train dataframe shape: {}".format(Train_df.shape))
# Train_df.head(5)
print("Test dataframe shape: {}".format(Test_df.shape))
# Test_df.head(5)

# %%



#* Loading img and Preprocessing
#* Numpy aray intersection.
class Npy_handler():
    # *Initializing class.
    def __init__(self):
        #! Warning: This is for testing
        # print("Npy_handler")
        #! Warning: This is for testing

        #* npy_path point to the npy dir.
        self.npy_path = "../input/reducing-image-sizes-to-32x32/"
        self.x_tr_npy, self.y_tr_npy, self.x_te_npy = self.load_npy()

    #! Warning: If you have time.
    #def get_dir(self):
    

    #* Numpy aray file loader.
    def load_npy(self):
        #! Warning: This is for testing
        # print("Npy_handler.load_npy")
        #! Warning: This is for testing
        
        if(self.npy_exist()):
            x_tr_npy = np.load(os.path.join(self.npy_path, "X_train.npy"))
            y_tr_npy = np.load(os.path.join(self.npy_path, "y_train.npy"))
            x_te_npy = np.load(os.path.join(self.npy_path, "X_test.npy"))
            return x_tr_npy, y_tr_npy, x_te_npy
        #? If the Numpy file does not exist.
        #? Return ["None"]
        else:
            x_tr_npy = np.array(["None"])
            y_tr_npy = np.array(["None"])
            x_te_npy = np.array(["None"])
            return x_tr_npy, y_tr_npy, x_te_npy
        
    #? Check the numpy file.
    def npy_exist(self):

        #! Warning: This is for testing
        # print("Npy_handler.npy_exist")
        #! Warning: This is for testing

        NPY_LIST = os.listdir(self.npy_path)
        #! Warning: Testing for load raw img 
        #! Warning: Don't forget to correct 'X_[test.npy' back to 'X_test.npy'
        if("X_test.npy" in NPY_LIST and
            'X_train.npy' in NPY_LIST and 'y_train.npy' in NPY_LIST):
            return True
        else:
            return False
    
    #* Return x_train, y_train, x_test
    def get_npy(self):
        #! Warning: This is for testing
        # print("Npy_handler.get_npy")
        #! Warning: This is for testing
        return self.x_tr_npy, self.y_tr_npy, self.x_te_npy
    
    #@ parameter x_tr, y_tr, x_te
    #@ Type np.array, np.array, np.array
    #* Save the file in default Dir path.
    def save_npy(self, x_tr, y_tr, x_te):
        #! Warning: This is for testing
        # print("Npy_handler.save_npy")
        #! Warning: This is for testing

        # print("save")
        #Save the img in numpy array
        np.save(os.path.join(self.npy_path,"x_train.npy"), x_tr)
        np.save(os.path.join(self.npy_path,"y_train.npy"), y_tr)
        np.save(os.path.join(self.npy_path,"x_test.npy"), x_te)


#! Warning: If you have time.
# def show_plot():
#     fig, ax = plt.subplots(9, 1, figsize=(180, 20))
#     ax[index].imshow(img)


#* Loading Img.jpg
class Img_Loader():
    #@ Parametr TrainDataFrame, TestDataFrame
    #@ Type pd.DataFrame, pd.DataFrame
    def __init__(self, train_df, test_df):
        #! Warning: This is for testing
        # print("Img_Loader")
        #! Warning: This is for testing
        self.INPUT_DIR = "../input/iwildcam-2019-fgvc6/"
        self.Train_df = train_df
        self.Test_df = test_df
        
        #Construct an Npy_handler to check the npy file.
        self.npy_handler = Npy_handler()
        #Construct an Preprocess_interface.
        self.preprocess_interface = Preprocess_interface()

    #* Return x_train, y_train, x_test
    def Get_the_data(self):


        #* Return: x_train, y_train, x_test  
        #* Back to Get_the_data()
        #? This function will call filter_name to get the file name from DataFrame.
        #? Get a list of [ LOAD_DIR, fname, label ]
        #? Then set the list into preprocess_interface.
        def load2process():
            #! Warning: This is for testing
            # print("Img_Loader.Get_the_data.load2process")
            #! Warning: This is for testing
            #* Get the list of [LOAD_DIR, fname, label]
            train2preprocess = filter_fname(train_or_not = True)
            #* Throw list to preprocess_interface.
            x_tr, y_tr = self.preprocess_interface.preprocess(train2preprocess,
                                                              train_or_not = True)
            #* Get a list of [LOAD_DIR, fname ]
            test2preprocess = filter_fname(train_or_not = False)
            #* Throw list to preprocess_interface.
            x_te = self.preprocess_interface.preprocess(test2preprocess,
                                                        train_or_not = False)
            
            #!Warning: For demo
            #!Warning: This is command will save the npy file. 
            # npy_handler.save_npy(x_tr, y_tr, x_te)
            
            return x_tr, y_tr, x_te    

        #@Parametr: train_or_not
        #@Type: boolean
        #* For check current file set is training or testing.
        #* It will return a list of [ LOAD_DIR, fname, label ]
        def filter_fname(train_or_not=True):
            #! Warning: This is for testing
            # print("Img_Loader.Get_the_data.filter_fname")
            #! Warning: This is for testing

            if train_or_not:

                LOAD_DIR = os.path.join(self.INPUT_DIR, "train_images")
                fname = Train_df['file_name'].values
                label = pd.get_dummies(Train_df['category_id']).values
                return [LOAD_DIR, fname, label]
            else:
                LOAD_DIR = os.path.join(self.INPUT_DIR, "test_images")
                fname = Test_df['file_name'].values
                return [LOAD_DIR, fname] 
            
        #! Warning: This is for testing
        # print("Img_Loader.Get_the_data")
        #! Warning: This is for testing

        #exist and return
        if (self.npy_handler).npy_exist():
            return self.npy_handler.get_npy()
        return load2process()
        
        
        

class Preprocess_interface():
    def __init__(self):
        #! This is for testing
        # print("Preprocess_interface")
        #! This is for testing
    #@ Parameter: preprocess_list, train_or_not
    #@ Type: list, boolean
    #? preprocess_list = [LOAD_DIR, fname, label(train) ]    
    #? train_or_not is to duplicate current file is training or testing.
    #* This function will return a np.array
    #* return shape: ( Total_of_file_numbers, width_in_pixel, height_in_pixel, color_chennel )
    def preprocess(self, preprocess_list=[], train_or_not=True ):
        #! Warning: This is for testing
        # print("Preprocess_interface.preprocess")
        #! Warning: This is for testing

        IMG_list = []
        LOAD_DIR = preprocess_list[0]
        fname = preprocess_list[1]
        
        #! Warning: Set the first 100th data to testing.
        for f in tqdm(fname[:100], desc="Loading_preprocess"):
            img_path = os.path.join(LOAD_DIR, f)
            IMG_list.append(self.Read2preprocess(img_path))
        
        if train_or_not:
            label = preprocess_list[2]
            #* Return x_train, y_train
            return np.stack(IMG_list), label[:100]
        else:
            #* Return x_test
            return np.stack(IMG_list)
        
    def Read2preprocess(self, IMG_PATH):
        #! Warning: This is for testing
        # print("Preprocess_interface.Read2preprocess")
        #! Warning: This is for testing
        def resize(img, default_size=32):
            #! Warning: This is for testing
            # print("Preprocess_interface.Read2preprocess.resize")
            #! Warning: This is for testing
            return cv2.resize(img, (32, )*2 ).astype('uint8')

        def normalize(img):
            #! Warning: This is for testing
            # print("Preprocess_interface.Read2preprocess.normalize")
            #! Warning: This is for testing
            img = img.astype('float32')
            return img / 255

        img = cv2.imread(IMG_PATH)
        img = resize(img)
        img = normalize(img)
        return img


IMG_loader = Img_Loader( Train_df, Test_df)
x_train, y_train, x_test = IMG_loader.Get_the_data()

# %% 
# *Data mining


# %%
# *Model construction
# *Using denseNet121

# @Set the include_top = False.
# ?We will define the transfer learning output.
# @Set the input_shape = [32, 32, 3]
# ?Input shape must fit out input dataset.
model_121 = DenseNet121(weights='imagenet')

# base_model = Sequential()
# base_model.add(model_121)
# base_model.add(GlobalAveragePooling2D())
# base_model.add(Dense())

# *check our CNN learning architecture
model_121.summary()



#%%
# !Testing VGG16
# from keras.applications.vgg16 import VGG16
# model_vgg16 = VGG16(weights='imagenet')
# model_vgg16.summary()

# %%
# *Data visualization
