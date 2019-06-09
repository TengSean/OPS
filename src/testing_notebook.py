# %%
# *Function library import
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

from sklearn.model_selection import train_test_split

import os
# import warning
import datetime
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

# %%
# *Data loading 


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
