# %%
# *Function library import
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.callbacks import Callback, ModelCheckpoint
import keras
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score



from tqdm import tqdm_notebook as tqdm


import os

# import warning
import datetime
import time

import json

import cv2
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
pd.options.display.max_columns = 999
import tensorflow as tf

INPUT_DIR = "../input/iwildcam-2019-fgvc6/"
# RESIZE_IMG_DIR = "../input/reducing-image-sizes-to-32x32/"
WEIGHTS_DIR = "../input/densenet-keras/"
# print(os.listdir('../input/reducingimagesizesto32x32/reducing-image-sizes-to-32x32/reducing-image-sizes-to-32x32'))
# print(os.listdir("../input/"))
print(os.listdir("../input/resize/resize/resize"))

# %%
# *Data csv loading 
Train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
Test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

print("Train dataframe shape: {}".format(Train_df.shape))
# Train_df.head(50)
# print("Test dataframe shape: {}".format(Test_df.shape))
# Test_df.head(5)

# %%

class Npy_handler():
#     NPY_PATH default point to the 
    def __init__(self):
#         print("Npy_handler")
#         npy_path point to the npy dir.
#         self.npy_path = "../input/reducing-image-sizes-to-32x32/"
        self.npy_path = "../input/resize/resize/resize"
        self.x_tr_npy, self.y_tr_npy, self.x_te_npy = self.load_npy()
        
#     def get_dir(self):
        
    def load_npy(self):
#         print("Npy_handler.load_npy")
        if(self.npy_exist()):
#             x_tr_npy = np.load(os.path.join(self.npy_path, "X_train.npy"))
#             y_tr_npy = np.load(os.path.join(self.npy_path, "y_train.npy"))
#             x_te_npy = np.load(os.path.join(self.npy_path, "X_test.npy"))
            x_tr_npy = np.load(os.path.join(self.npy_path, "x_train.npy"))
            y_tr_npy = np.load(os.path.join(self.npy_path, "y_train.npy"))
            x_te_npy = np.load(os.path.join(self.npy_path, "x_test.npy"))
            return x_tr_npy, y_tr_npy, x_te_npy
        else:
            x_tr_npy = np.array(["None"])
            y_tr_npy = np.array(["None"])
            x_te_npy = np.array(["None"])
            return x_tr_npy, y_tr_npy, x_te_npy
        
    def npy_exist(self):
#         print("Npy_handler.npy_exist")
        NPY_LIST = os.listdir(self.npy_path)
#         if("X_test.npy" in NPY_LIST and
#             'X_train.npy' in NPY_LIST and 'y_train.npy' in NPY_LIST):
#             return True
        if("x_test.npy" in NPY_LIST and
            'x_train.npy' in NPY_LIST and 'y_train.npy' in NPY_LIST):
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



# def show_plot():
#     fig, ax = plt.subplots(9, 1, figsize=(180, 20))
#     ax[index].imshow(img)

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
                print("[TRACE]: Preprocess_interface.Read2preprocess.resize(DEFAULT)")
                print("[INFO]: Input dim: {}".format((default_size, default_size)))
                return cv2.resize(img, (default_size, )*2 ).astype('uint8')
            else:
                print("[TRACE]: Preprocess_interface.Read2preprocess.resize(DIMENTION)")
                print("[INFO]: Input dim: {}".format(Dim))
                return cv2.resize(img, Dim ).astype('uint8')
                
        def normalize(img):
#             print("Preprocess_interface.Read2preprocess.normalize")
            img = img.astype('float32')
            return img / 255
        img = cv2.imread(str(IMG_PATH))
        img = resize(img, default_size, Dimention)

        img = normalize(img)
        return img

# %%
%%time


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

# IMG_loader = Img_Loader( Train_df, Test_df)
# x_train, y_train, x_test = IMG_loader.Get_the_data()


# fold_list = IMG_loader.validation_sperator(5)

# for tr_index, te_index in fold_list:
#     gen = IMG_loader.Load_batch('Train', Train_df['file_name'][tr_index],
#                           Train_df['category_id'][tr_index], batch_size=250)
#     gen = IMG_loader.Load_batch('Test', Test_df['file_name'][te_index],
#                         batch_size=250)
#     while True:
#         print('cc')
#          get_next = next(gen)
#         print(get_next[0].shape)
#         print(get_next[1].shape)
#     print(label.shape)

# %%

# %%
# *Model construction
# *Using denseNet121

# # os.listdir(WEIGHTS_DIR)
# WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'DenseNet-BC-121-32-no-top.h5')
# # # WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, 'DenseNet-BC-121-32.h5')
# # # WEIGHTS_PATH

# # # @Set the include_top = False.
# # # ?We will define the transfer learning output.
# # # @Set the input_shape = [32, 32, 3]
# # # ?Input shape must fit out input dataset.
# model_121 = DenseNet121(weights= WEIGHTS_PATH,
#                         include_top=False,
#                         input_shape=(32,32,3))

# base_model = Sequential()
# base_model.add(model_121)
# base_model.add(GlobalAveragePooling2D())
# # # Testing for output 14 class.
# # # But here we got a question.
# # # There are 23 class(1 empty), if we set the output dimention is 24 what will happen?
# base_model.add(Dense(len(total_category), activation="softmax"))
# # # check our CNN learning architecture
# base_model.summary()



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
        # ?Do something here...
        # Get the validation set
        # for loop start to train

#         self.Build_model(in_shape=(256,256,3))


        fold_list = self.IMG_loader.validation_sperator(3)
        for round, (tr_index, val_index) in enumerate(fold_list):
            print("Fold :{}".format(round + 1))
            print("train: {}, test: {}".format(len(tr_index), len(val_index)))

            x_train = Train_df['file_name'][tr_index]
            y_train = Train_df['category_id'][tr_index]
            print("train: {}".format(x_train))            
            x_val = Train_df['file_name'][val_index]
            y_val = Train_df['category_id'][val_index]
            print("test: {}".format(x_val.shape))
            Train_gen = self.IMG_loader.Load_batch('Train', x_train, y_train, batch_size=self.BATCH_SIZE)
            Val_gen = self.IMG_loader.Load_batch('Train', x_val, y_val, batch_size=self.BATCH_SIZE)
            model_name = "model_fold-"+ str(round + 1) + ".h5"
            
            checkpoint = ModelCheckpoint(model_name, 
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
        return self.result
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
            validation_split=0.33
        )
        return self.result
#     def demo_predict(self):
    def write_json(self):
        history_df = self.result.history
        with open('history.json', 'w') as f:
            json.dump(history_df, f)

#     def show_img(self):


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
history = test_transfer.Train_model_batch()
# history = test_transfer.Train_model_all()

# %%
class Img_DEMO():
    def __init__(self, Train_df, Test_df):
        
        self.INPUT_DIR = "../input/iwildcam-2019-fgvc6/"
        tra_file_path = Train_df['file_name'][:16]
        te_file_path = Test_df['file_name'][:16]
        self.Tra_labels  = Train_df['category_id'][:16]
        self.Tra_FILE_PATH = tra_file_path.apply(lambda x: os.path.join(os.path.join(self.INPUT_DIR
                                                                    ,"train_images"), x))

        self.Te_FILE_PATH = te_file_path.apply(lambda x: os.path.join(os.path.join(self.INPUT_DIR
                                                                    ,"test_images"), x))

    def show_Img(self):
        
        def resize(img):
            return cv2.resize(img, (32, )*2 ).astype('uint8')
        def normalize(img):
            img = img.astype('float32')
            return img / 255
        Row = 4
        Col = 4
        file_name=["raw_img.jpg", "resize_img.jpg", "nor_img.jpg"]
        for round in range(3):
            
            plt.figure()
            fig, ax = plt.subplots(4, 4, figsize=(Row* 5,Col*5))
            for row in range(Row):
                for col in range(Col):
                    if round == 0:
                        ax[row, col].imshow(cv2.imread(self.Tra_FILE_PATH[row * 4 + col]))
                    elif round == 1:
                        ax[row, col].imshow(resize(cv2.imread(self.Tra_FILE_PATH[row * 4 + col])))
                    elif round == 2:
                        img_tmp = resize(cv2.imread(self.Tra_FILE_PATH[row * 4 + col]))
                        ax[row, col].imshow(normalize(img_tmp))
        #         plt.imshow(img)
            # plt.imshow(x_train_img[0])
            plt.savefig(file_name[round])
            plt.show()
        
class predict_DEMO():
    def __init__(self):
        self.model = tf.contrib.keras.models.load_model("../input/output/model.h5")
#         self.model.summary()
        self.preprocess_interface = Preprocess_interface()
        self.Species_dict = {0: ['other'], 1: ['deer'], 2: ['moose'], 3: ['squirrel'], 4: ['rodent'],5: ['small_mammal'],
                6: ['elk'],7: ['pronghorn_antelope'],8: ['rabbit'],9: ['bighorn_sheep'],10: ['fox'],
                11: ['coyote'],12: ['black_bear'],13: ['raccoon'],14: ['skunk'],15: ['wolf'],16: ['bobcat'],
                17: ['cat'],18: ['dog'],19: ['opossum'],20: ['bison'],21: ['mountain_goat'], 22: ['mountain_lion']}
        
        self.Species_df = pd.DataFrame(self.Species_dict)
        self.history = self.read_resultJson()
        
    def read_resultJson(self):
        with open('../input/output/history.json' , 'r') as reader:
            history = json.loads(reader.read())
        return history
    
    def predict(self, path):
        result_img = self.preprocess_interface.Read2preprocess(path, 32)
#         print(result_img.shape)
        y_test = self.model.predict(result_img[None,:])
        y_test = y_test.argmax(axis=1)
        print("Predict result: {}".format(self.Species_df[y_test]))
        
    def show_category(self):
        print(self.Species_df)
        
    def show_resultJson(self):
        h_df = pd.DataFrame(self.history)
        epochs = range(len(h_df['loss']))
        plt.figure()
        fig, ax = plt.subplots(1,2,figsize=(18,4))
        ax[0].plot(epochs, h_df['loss'], label='Training loss')
        ax[0].plot(epochs, h_df['val_loss'], label='Validation loss')
        ax[0].set_title('Training and validation loss')
        ax[0].legend()
        ax[1].plot(epochs, h_df['acc'],label='Training accuracy')
        ax[1].plot(epochs, h_df['val_acc'], label='Validation accuracy')
        ax[1].set_title('Training and validation accuracy')
        ax[1].legend()
        plt.show()
        

img_demo= Img_DEMO(Train_df, Test_df)
img_demo.show_Img()

# predict_demo = predict_DEMO()
predict_demo.predict("../input/test-folder/test1.jpg")
predict_demo.show_category()
predict_demo.show_resultJson()