##import statements
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
import CustomDataGenerator
from tensorflow.keras.layers import Input
import os,time,math,sys
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import AlexNet
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121

from collections import OrderedDict
from sklearn import metrics

##define macros
BATCH_SIZE = 64
GPU_ID = "2"
GPU_FRACTION = 0.9


##set GPU options
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_ID
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
k.set_session(tf.Session(config=config))


##utility function to fetch filename of checkpoint in a directory
def get_filename(dir):
	(_, _, filenames) = next(os.walk(dir))
	for i in filenames:
		if(i.find('data')!=-1):
			data = i
			break
	return(data.split('.')[0])


##create data generator for test
test_path = "/home/vedantb/datav5/Test_v53_3c_by5.txt"
test_df = pd.read_csv(test_path,sep=".png ",header=None)
test_df[0] = test_df[0].astype(str)+'.png'
test_df[0]=test_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)
test_list_IDs = test_df[0].values
test_labels = test_df[1].values
test_gen = CustomDataGenerator.DataGenerator(test_list_IDs,test_labels,batch_size=BATCH_SIZE,n_channels=1,shuffle=False)


##create data generator for val
val_path = "/home/vedantb/datav5/Val_v53_3c_by5.txt"
val_df = pd.read_csv(val_path,sep=".png ",header=None)
val_df[0] = val_df[0].astype(str)+'.png'
val_df[0]=val_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)
val_list_IDs = val_df[0].values
val_labels = val_df[1].values
val_gen = CustomDataGenerator.DataGenerator(val_list_IDs,val_labels,batch_size=BATCH_SIZE,n_channels=1,shuffle=False)

##define model callbacks
def get_callbacks(name):
	model_ckpt = keras.callbacks.ModelCheckpoint("./callback_data/"+name+"-ckpt-{epoch:02d}", monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto', period=1)
	tensorb = keras.callbacks.TensorBoard(log_dir='./callback_data/'+name+'-logs', histogram_freq=0, batch_size=16, write_graph=True, write_grads=False, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
	csv_logg = keras.callbacks.CSVLogger("./callback_data/" +name+"-csv", separator=',', append=True)
	prog_bar =keras.callbacks.ProgbarLogger(count_mode='samples', stateful_metrics=None)
	callbacks  = [prog_bar,model_ckpt,tensorb,csv_logg]
	return(callbacks)

##define models dictionary
models = {}
#models['resnet50_2']=ResNet50
#models['mobilenet'] = MobileNetV2
#models['alexnet']=AlexNet
#models['vgg16']=VGG16
models['densenet121'] = DenseNet121
#models['xception']=Xception


##import model
def get_model(name):
	model=models[name]
	if(name!="alexnet"):
		model = model(weights=None,input_shape=(240,320,1),include_top=True,classes=3)
	else:
		model=model.alexnet_model()
	checkpoint_path = get_filename('./best_models/'+name)
	model.load_weights('./best_models/{0}/{1}'.format(name,checkpoint_path))
	model.compile(loss='categorical_crossentropy',
		optimizer=keras.optimizers.Adam(lr=0.00001),
		metrics=['accuracy'])
	layer_to_analyze = [i.name for i in model.layers][-3]
	return(model,layer_to_analyze)


##compute inference time, val accuracies, test accuracies, store predictions
def analysis(name):
	t1 = time.time()
	model,_ = get_model(name)
	t2 = time.time()
	
	num_samples_test = int(BATCH_SIZE*(test_labels.shape[0]//BATCH_SIZE))

	y_pred_test=model.predict_generator(test_gen,verbose=1,use_multiprocessing=True)
	t3 = time.time()
	test_confusion_matrix = metrics.confusion_matrix(test_labels[:num_samples_test], y_pred_test.argmax(axis=1))
	test_three_class_accuracy = (test_confusion_matrix[0,0]+test_confusion_matrix[1,1]+test_confusion_matrix[2,2])/np.sum(test_confusion_matrix)
	test_two_class_accuracy = (test_confusion_matrix[0,0]+test_confusion_matrix[2,2])/(test_confusion_matrix[0,0]+test_confusion_matrix[2,0]+test_confusion_matrix[0,2]+test_confusion_matrix[2,2])

	t4 = time.time()
	y_pred_val=model.predict_generator(val_gen,verbose=1,use_multiprocessing=True)
	t5 = time.time()
	num_samples_val = int(BATCH_SIZE*(val_labels.shape[0]//BATCH_SIZE))
	val_confusion_matrix = metrics.confusion_matrix(val_labels[:num_samples_val], y_pred_val.argmax(axis=1))
	val_three_class_accuracy = (val_confusion_matrix[0,0]+val_confusion_matrix[1,1]+val_confusion_matrix[2,2])/np.sum(val_confusion_matrix)
	val_two_class_accuracy = (val_confusion_matrix[0,0]+val_confusion_matrix[2,2])/(val_confusion_matrix[0,0]+val_confusion_matrix[2,0]+val_confusion_matrix[0,2]+val_confusion_matrix[2,2])

	records = OrderedDict()
	records['batch_size_generators'] = BATCH_SIZE
	records['GPU_ID'] = GPU_ID
	records['GPU_FRACTION'] = GPU_FRACTION
	records['time_to_load'] = t2-t1
	records['test_inference_time'] = t3-t2
	records['val_inference_time'] = t5-t4
	records['test_confusion_matrix'] = test_confusion_matrix
	records['test_two_class_accuracy'] = test_two_class_accuracy
	records['test_three_class_accuracy'] = test_three_class_accuracy
	records['val_confusion_matrix'] = val_confusion_matrix
	records['val_two_class_accuracy'] = val_two_class_accuracy
	records['val_three_class_accuracy'] = val_three_class_accuracy

	return(records,y_pred_val,y_pred_test)

##run code and save files
for name,m in models.items():
	record,pred_val,pred_test=analysis(name)
	np.save('./best_models/{0}/records'.format(name),record)
	np.save('./best_models/{0}/pred_val'.format(name),pred_val)
	np.save('./best_models/{0}/pred_test'.format(name),pred_test)	

	
