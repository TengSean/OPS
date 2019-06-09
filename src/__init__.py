#%%
if __name__ == "__main__":
    print("hello world")

#%%
class Transfer_learn():
    def __init__(self):
        # ?Do something here...
        self.model = self.Build_model()

    # *activation = Soft_max, for classification in last layer(output layer).
    # *loss = catagory_corssentropy
    # *Optimizier = adam
    def Build_model(self):
        # ?Do something here...
        
        # *Loadign VGG16 or denseNet121
        def load_pretrain_model():
            # ?Do something here...

        # *Custome Dense function.
        # *This integrate some CNN layer.
        # *Such as droupout, loss and activation.
        def dense(self):
            # ?Do something here...


    def Train_model(self):
        # ?Do something here...
        

    # *Sean prefer to 5-Fold cross-validation
    def Validataion(self):
        # ?Do something here...
        # TODO: If tou have time.
        # !def fit_generator(self, generator, samples_per_epoch, nb_epoch,
        # !              verbose=1, callbacks=[],
        # !              validation_data=None, nb_val_samples=None,
        # !              class_weight=None, max_q_size=10, **kwargs):

class Pre_process():
    # *Initializing private value
    def __init__(self):
        # ?Do something here...
    
    # *Set the img into 0~1 and reshaping to match the input layer.
    def normalize_img(self):
        # ?Do something here...

        # TODO: If you have time.
        def resize_img(self):
            # ?Do something here...
    

    # *Seperate training and val dataset.
    def random_sampling(self):
        # ?Do something here...

# TODO: If you have time.
class Draw_plt():

    def __init__():
#%%


#%%
