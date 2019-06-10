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
print(os.listdir('../input'))
# %%

print("csv: {}".format(os.listdir(RAW_INPUT_DIR)))
print("Resize: {}".format(os.listdir(RESIZE_INPUT_DIR)))
print("Model: {}".format(os.listdir((MODEL_WEIGHTS_DIR))))
# %%
# *Data loading 
Train_df = pd.read_csv(RAW_INPUT_DIR + '/train.csv')
Test_df = pd.read_csv(RAW_INPUT_DIR + '/test.csv')

# print("Train dataframe shape: {}".format(Train_df.shape))
# Train_df.head(5)
# print("Test dataframe shape: {}".format(Test_df.shape))
# Test_df.head(5)

# %%

# def show_plot():
#     fig, ax = plt.subplots(9, 1, figsize=(180, 20))
#     ax[index].imshow(img)


def Read2process(IMG_PATH=None):
    def resize(img, default_size=32):
        return cv2.resize(img, (32, )*2 ).astype('uint8')
    def normalize(img):
        img = img.astype('float32')
        return img / 255
    img = cv2.imread(IMG_PATH)
    img = resize(img)
    img = normalize(img)
    return img

def Get_the_data(train_or_not=True):
    IMG_list = []

    fname = []
    label = []
    if train_or_not:
        LOAD_DIR = os.path.join(IMG_DIR, "train_images")
        fname = Train_df['file_name'].values
        label = pd.get_dummies(Train_df['category_id']).values
    else:
        LOAD_DIR = os.path.join(IMG_DIR, "test_images")
        fname = Test_df['file_name'].values
#     print(LOAD_DIR)

#     for f, l in tqdm(zip(fname[:100], label[:100]), desc="Loading and resizing"):
    for f, l in tqdm(zip(fname[:100], label[:100])):
        img_path = os.path.join(LOAD_DIR, f)
        IMG_list.append(Read2process(img_path))
    if train_or_not:
#           np.save("x_train.npy", np.stack(IMG_list))
#           np.save("y_train.npy", label[:100])
        return np.stack(IMG_list), label[:100]
    else:
#           np.save("x_test.npy", np.stack(IMG_list))
        return np.stack(IMG_list)

im, la = Get_the_data()
print(im.shape)
print(la.shape)
# %% 
# *Data mining


# %%
# *Model construction
# *Using denseNet121

# @Set the include_top = False.
# ?We will define the transfer learning output.
# @Set the input_shape = [32, 32, 3]
# ?Input shape must fit out input dataset.
model_121 = DenseNet121(weights='imagenet',
                        include_top=False,
                        input_shape=(32,32,3))

base_model = Sequential()
base_model.add(model_121)
base_model.add(GlobalAveragePooling2D())
# base_model.add(Dense())

# *check our CNN learning architecture
# model_121.summary()



#%%
# !Testing VGG16
# from keras.applications.vgg16 import VGG16
# model_vgg16 = VGG16(weights='imagenet')
# model_vgg16.summary()

# %%
# *Data visualization
