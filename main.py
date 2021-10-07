import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils.data_utils import Sequence
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import RMSprop as rmsprop
from tensorflow.keras import backend as k
import os


# from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import AlexNet
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.xception import Xception
from collections import OrderedDict

import CustomDataGenerator
from tensorflow.keras.layers import Input

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="2"

train_path = "/home/vedantb/datav5/Train_v53_3c_by5.txt"
val_path = train_path.replace("Train","Val")


config = tf.ConfigProto()
 

config.gpu_options.per_process_gpu_memory_fraction = 0.9

k.set_session(tf.Session(config=config))





train_df = pd.read_csv(train_path,sep=".png ",header=None)
val_df = pd.read_csv(val_path,sep=".png",header=None)
train_df[0] = train_df[0].astype(str)+'.png'
train_df[0]=train_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)
val_df[0] = val_df[0].astype(str)+'.png'
train_df[1] = train_df[1].astype(str)
val_df[0]=val_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)

val_df[1] = val_df[1].astype(str)



train_list_IDs = train_df[0].values
train_labels = train_df[1].values
train_gen = CustomDataGenerator.DataGenerator(train_list_IDs,train_labels,n_channels=1)
val_list_IDs = val_df[0].values
val_labels = val_df[1].values
val_gen = CustomDataGenerator.DataGenerator(val_list_IDs,val_labels,n_channels=1)


models = OrderedDict()
#models['mobilenet'] = MobileNetV2
# models['alexnet']=AlexNet
# models['vgg16']=VGG16
# models['xception']=Xception
models['densenet121'] = DenseNet121
# this could also be the output a different Keras model or layer
input_tensor = Input(shape=(240, 320, 1))  # this assumes K.image_data_format() == 'channels_last'

# model=(include_top=True, weights='imagenet')
# model=VGG16(weights=None,input_shape=(240,320,1),include_top=True,classes=3)
def get_callbacks(name):

	model_ckpt = keras.callbacks.ModelCheckpoint("./callback_data_sgd/"+name+"-ckpt-{epoch:02d}", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
	early_stop = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.0025, patience=1, verbose=1, mode='auto', baseline=.90, restore_best_weights=False)
	tensorb = keras.callbacks.TensorBoard(log_dir='./callback_data_sgd/'+name+'-logs', histogram_freq=0, batch_size=16, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
	csv_logg = keras.callbacks.CSVLogger("./callback_data_sgd/" +name+"-csv", separator=',', append=True)
	prog_bar =keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
	callbacks  = [prog_bar,model_ckpt,tensorb,csv_logg]
	return(callbacks)
			
histories = {}
def run(models):
	for i,j in models.items():
		print(i,j)
		if(i!="alexnet"):
			model = j(weights=None,input_shape=(240,320,1),include_top=True,classes=3)
		else:
			model=j.alexnet_model()
		model.compile(loss='categorical_crossentropy',
			optimizer=keras.optimizers.Adam(lr=0.0001),
			metrics=['accuracy'])
		callbacks=get_callbacks(i)
		history = model.fit_generator(generator=train_gen,callbacks=callbacks,validation_data=val_gen,epochs=50,use_multiprocessing=True)
		histories[i]=history
	
run(models)
import pickle
with open('hist-121.pk','wb') as f:
	pickle.dump(histories,f)



