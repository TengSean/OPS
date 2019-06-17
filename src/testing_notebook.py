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
#     NPY_PATH default point to the 
    def __init__(self):
#         print("Npy_handler")
#         npy_path point to the npy dir.
        self.npy_path = "../input/reducing-image-sizes-to-32x32/"
        self.x_tr_npy, self.y_tr_npy, self.x_te_npy = self.load_npy()
        
#     def get_dir(self):
        
    def load_npy(self):
#         print("Npy_handler.load_npy")
        if(self.npy_exist()):
            x_tr_npy = np.load(os.path.join(self.npy_path, "X_train.npy"))
            y_tr_npy = np.load(os.path.join(self.npy_path, "y_train.npy"))
            x_te_npy = np.load(os.path.join(self.npy_path, "X_test.npy"))
            return x_tr_npy, y_tr_npy, x_te_npy
        else:
            x_tr_npy = np.array(["None"])
            y_tr_npy = np.array(["None"])
            x_te_npy = np.array(["None"])
            return x_tr_npy, y_tr_npy, x_te_npy
        
    def npy_exist(self):
#         print("Npy_handler.npy_exist")
        NPY_LIST = os.listdir(self.npy_path)
        if("X_test.npy" in NPY_LIST and
            'X_train.npy' in NPY_LIST and 'y_train.npy' in NPY_LIST):
            return True
        else:
            return False
        
    def get_npy(self):
#         print("Npy_handler.get_npy")
        return self.x_tr_npy, self.y_tr_npy, self.x_te_npy
    
    def save_npy(self, x_tr, y_tr, x_te):
#         print("Npy_handler.save_npy")
        print("save")
#         Testing for TengSean
#         Save the img in numpy array
#         np.save(os.path.join(self.npy_path,"x_train.npy"), x_tr)
#         np.save(os.path.join(self.npy_path,"y_train.npy"), y_tr)
#         np.save(os.path.join(self.npy_path,"x_test.npy"), x_te)
        
        # Testing for other member when using kaggle kernel
        np.save("x_train.npy", x_tr)
        np.save("y_train.npy", y_tr)
        np.save("x_test.npy", x_te)


#! Warning: If you have time.
# def show_plot():
#     fig, ax = plt.subplots(9, 1, figsize=(180, 20))
#     ax[index].imshow(img)

# %%
#* Loading Img.jpg
class Img_Loader():
    #Construct arg1: TrainDataFrame, arg2: TestDataFrame
    def __init__(self, train_df, test_df):
#         print("Img_Loader")
        self.INPUT_DIR = "../input/iwildcam-2019-fgvc6/"
        self.Train_df = Train_df[['category_id','file_name']]
        self.Test_df = Test_df[['file_name']]
        
        #Construct an Npy_handler to check the npy file.
        self.npy_handler = Npy_handler()
        #Construct an Preprocess_interface.
        self.preprocess_interface = Preprocess_interface()

    def Get_the_data(self):

        def load2process():
#             print("Img_Loader.Get_the_data.load2process")

            # [LOAD_DIR, fname ]
            test2preprocess = filter_fname(train_or_not = False)
            # Throw [LOAD_DIR, fname ] to preprocess_interface.
            x_te = self.preprocess_interface.preprocess(test2preprocess,
                                                        train_or_not = False)
            # [LOAD_DIR, fname, label]
            train2preprocess = filter_fname(train_or_not = True)
            #Throw [LOAD_DIR, fname, label] to preprocess_interface.
            x_tr, y_tr = self.preprocess_interface.preprocess(train2preprocess,
                                                              train_or_not = True)

#             self.npy_handler.save_npy(x_tr, y_tr, x_te)
            
            return x_tr, y_tr, x_te    

        def filter_fname(train_or_not=True):
#             print("Img_Loader.Get_the_data.filter_fname")
            if train_or_not:
                LOAD_DIR = os.path.join(self.INPUT_DIR, "train_images")
                fname = Train_df['file_name'].values
                label = pd.get_dummies(Train_df['category_id']).values
                return [LOAD_DIR, fname, label]
            else:
                LOAD_DIR = os.path.join(self.INPUT_DIR, "test_images")
                fname = Test_df['file_name'].values
                return [LOAD_DIR, fname] 
            

#         print("Img_Loader.Get_the_data")
        # If exist then return
        if (self.npy_handler).npy_exist():
            # Here return x_train, y_train, x_test
            return self.npy_handler.get_npy()
        return load2process()
        
        
        
        
        
    # This is for generator function.
    # This function use for fit_generation() function.
    # Load batch from x_file_path, y_label
    def Load_batch(self, mode="Train", Fpath=None, Labels=None, batch_size=70):
        
#         print("[TRACE]: Img_loader.Load_batch")
#         print(Labels)

        file_mode = "train_images" if mode=='Train' else "test_images"
        # Get the complete file dir.
        FILE_PATH = Fpath.apply(lambda x: os.path.join(os.path.join(self.INPUT_DIR
                                                                    ,file_mode), x))
#         FILE_PATH = Fpath.apply(lambda x: print(x))

        while True:
            print("[TRACE]: Img_loader.Load_batch")
#             yield Load_training(FILE_PATH, batch_size) if mode == 'Train' else Load_testing(FILE_PATH, batch_size)
            Img_list = []
            Labels_list = []
            if mode=="Train":
                print("[TRACE]: Img_loader.Load_batch.Load_training")
                Labels = pd.get_dummies(Labels).values
                for (fpath, label) in zip(FILE_PATH, Labels):
    #                 print("file path: {}, label: {}".format(fpath, label))
#                     img = self.preprocess_interface.Read2preprocess(IMG_PATH=fpath, Dimention=(1024, 747))
                    img = self.preprocess_interface.Read2preprocess(IMG_PATH=fpath, Dimention=(256, 256))
                    Img_list.append(img)
                    Labels_list.append(label)
                    if len(Img_list) == batch_size :
                        return_Img = np.stack(Img_list)
                        return_labels = np.stack(Labels_list)
    #                     print("[TRACE]: {}".format(return_Img.shape))
    #                     print("[TRACE]: {}".format(return_labels.shape))
                        Img_list = []
                        Labels_list = []
                        yield (return_Img, return_labels)
            elif mode == "Test":
                print("[TRACE]: Img_loader.Load_batch.Load_testing")
                for (fpath, label) in zip(FILE_PATH, Labels):
    #                 print("file path: {}, label: {}".format(fpath, label))
#                     img = self.preprocess_interface.Read2preprocess(IMG_PATH=fpath, Dimention=(1024, 747))
                    img = self.preprocess_interface.Read2preprocess(IMG_PATH=fpath, Dimention=(256, 256))
                    Img_list.append(img)
                    if len(Img_list) == batch_size :
                        return_Img = np.stack(Img_list)
    #                     print("[TRACE]: {}".format(return_Img.shape))
                        Img_list = []
                        yield return_Img
            else:
                print("[Error]: Wrong input mode!")
            
    
    def validation_sperator(self, Nfold=5):
        sfold = StratifiedKFold(n_splits = Nfold)        
        fi = np.array(Train_df['file_name'][:,None])
        la = np.array(Train_df['category_id'][:,None])
        return sfold.split(fi, la)
        
        
        
# %%
class Preprocess_interface():
    def __init__(self):
        self.a=1
#         print("Preprocess_interface")
    # preprocess_list = [LOAD_DIR, fname, label(train) ]    
    def preprocess(self, preprocess_list=[], train_or_not=True ):
#         print("[TRACE]: Preprocess_interface.preprocess")
        
        IMG_list = []
        LOAD_DIR = preprocess_list[0]
        fname = preprocess_list[1]
        
        # Set the first 100th data to testing.
        for f in tqdm(fname, desc="Loading_preprocess"):
            img_path = os.path.join(LOAD_DIR, f)
            IMG_list.append(self.Read2preprocess(img_path, default_size = 32, Dimention=None))
        
        if train_or_not:
            label = preprocess_list[2]
            #  Here return x_tr, y_tr
            return np.stack(IMG_list), label
        else:
            #  Here return x_te
            return np.stack(IMG_list)
        
    def Read2preprocess(self, IMG_PATH="", default_size = 32, Dimention=None):
#         print("[TRACE]: Preprocess_interface.Read2preprocess")

        def resize(img, default_size=32, Dim=None):
            if Dimention == None:
#                 print("[TRACE]: Preprocess_interface.Read2preprocess.resize(DEFAULT)")
                return cv2.resize(img, (default_size, )*2 ).astype('uint8')
            else:
#                 print("[TRACE]: Preprocess_interface.Read2preprocess.resize(DIMENTION)")
                return cv2.resize(img, Dim ).astype('uint8')
                
        def normalize(img):
#             print("Preprocess_interface.Read2preprocess.normalize")
            img = img.astype('float32')
            return img / 255
        img = cv2.imread(str(IMG_PATH))
        img = resize(img, default_size, Dimention)

        img = normalize(img)
        return img


# IMG_loader = Img_Loader( Train_df, Test_df)
# x_train, y_train, x_test = IMG_loader.Get_the_data()

# %% 
# *Data mining


# %%
# *Model construction
# *Using denseNet121

# @Set the include_top = False.
# ?We will define the transfer learning output.
# @Set the input_shape = [32, 32, 3]
# ?Input shape must fit out input dataset.
# model_121 = DenseNet121(weights='imagenet')

# base_model = Sequential()
# base_model.add(model_121)
# base_model.add(GlobalAveragePooling2D())
# base_model.add(Dense())

# *check our CNN learning architecture
# model_121.summary()

class Transfer_learn():
    def __init__(self, ):
        # ?Do something here...
        # Get the data.
        # self.IMG_loader = Img_Loader( Train_df, Test_df)
        # self.x_train, self.y_train, self.x_test = IMG_loader.Get_the_data()
        self.result = []
#         self.NUM_EPOCHS = 5
#         self.BATCH_SIZE = 30
        
        self.NUM_EPOCHS = 35
        self.BATCH_SIZE = 64
        
#         self.model = self.Build_model()
        self.IMG_loader = Img_Loader( Train_df, Test_df)
    # *activation = Soft_max, for classification in last layer(output layer).
    # *loss = catagory_corssentropy
    # *Optimizier = adam
    def Build_model(self, in_shape):
        # ?Do something here...
        print("[TRACE]: Transfer_learn.Build_model")
        # *Loadign VGG16 or denseNet121
        def load_pretrain_model(in_shape):
            # ?Do something here...
            return DenseNet121(weights= os.path.join(WEIGHTS_DIR, 'DenseNet-BC-121-32-no-top.h5'),
                                        include_top=False,
                                        input_shape=in_shape)
        # *Custome Dense function.
        # *This integrate some CNN layer.
        # *Such as droupout, loss and activation.
#         def dense(self):
            # ?Do something here...
            
        print("[INFO]: Starting to build the model")
        model_121 = load_pretrain_model(in_shape)

        self.model = Sequential()
        self.model.add(model_121)
        self.model.add(GlobalAveragePooling2D())

        
        # # Testing for output 14 class.
        # # But here we got a question.
        # # There are 23 class(1 empty), if we set the output dimention is 24 what will happen?
        self.model.add(Dense(14, activation="softmax"))
        print("[INFO]: Finish the model")
        # # check our CNN learning architecture
#         base_model.summary()
        (self.model).compile(loss='categorical_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy'])
    
            
            
    def Train_model_batch(self):
#         # ?Do something here...
#         # Get the validation set
#         # for loop start to train

        self.Build_model(in_shape=(256,256,3))


        fold_list = self.IMG_loader.validation_sperator(3)
        for round, (tr_index, val_index) in enumerate(fold_list):
            print("Fold :{}".format(round + 1))
            x_train = Train_df['file_name'][tr_index]
            y_train = Train_df['category_id'][tr_index]
            
            x_val = Train_df['file_name'][val_index]
            y_val = Train_df['category_id'][val_index]
            
            Train_gen = self.IMG_loader.Load_batch('Train', x_train, y_train, batch_size=self.BATCH_SIZE)
            Val_gen = self.IMG_loader.Load_batch('Train', x_val, y_val, batch_size=self.BATCH_SIZE)
            
            checkpoint = ModelCheckpoint('model_fold-{}.h5'.format(round+1), 
                                monitor='val_acc', 
                                verbose=1, 
                                save_best_only=True, 
                                save_weights_only=False,
                                mode='auto')
            
            
            history = (self.model).fit_generator(
                                            Train_gen,
                                            steps_per_epoch=len(tr_index) // self.BATCH_SIZE,
                                            callbacks=[checkpoint],
                                            validation_data=Val_gen,
                                            validation_steps=len(val_index) // self.BATCH_SIZE,
                                            epochs=self.NUM_EPOCHS)
            self.result.append(history)
            
    def Train_model_all(self):
        
        self.Build_model(in_shape=(32, 32, 3))
        
        checkpoint = ModelCheckpoint('model.h5', 
                    monitor='val_acc', 
                    verbose=1, 
                    save_best_only=True, 
                    save_weights_only=False,
                    mode='auto')

        x_train, y_train, x_test = self.IMG_loader.Get_the_data()
        
        
        self.result = (self.model).fit(
            x=x_train,
            y=y_train,
            batch_size=self.BATCH_SIZE,
            epochs=self.NUM_EPOCHS,
            callbacks=[checkpoint],
            validation_split=0.1
        )
        
        
        
#     # *Sean prefer to 5-Fold cross-validation
#     def Validataion(self):
#         # ?Do something here...
#         # TODO: If tou have time.
#         # !def fit_generator(self, generator, samples_per_epoch, nb_epoch,
#         # !              verbose=1, callbacks=[],
#         # !              validation_data=None, nb_val_samples=None,
#         # !              class_weight=None, max_q_size=10, **kwargs):


test_transfer = Transfer_learn()
# test_transfer.Build_model()
test_transfer.Train_model_all()

#%%
# !Testing VGG16
# from keras.applications.vgg16 import VGG16
# model_vgg16 = VGG16(weights='imagenet')
# model_vgg16.summary()

# %%
# *Data visualization
