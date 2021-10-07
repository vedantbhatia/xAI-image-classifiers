import shap
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
# from tensorflow.keras import backend as K
import CustomDataGenerator
from tensorflow.keras.layers import Input
import os,time,math,sys
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
import AlexNet
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.densenet import DenseNet169
from collections import OrderedDict
from sklearn import metrics
from tensorflow.keras.preprocessing import image as Image
import cv2

BATCH_SIZE = 8
GPU_ID = "2"
GPU_FRACTION = 0.9


##set GPU options
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_ID
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = GPU_FRACTION
k.set_session(tf.Session(config=config))


def get_filename(dir):
    (_, _, filenames) = next(os.walk(dir))
    for i in filenames:
        if(i.find('data')!=-1):
            data = i
            break
    return(data.split('.')[0])

def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = Image.load_img(path, target_size=(240, 320))
    if preprocess:
        x = Image.img_to_array(x)
        x = x/255.0
        x = np.expand_dims(x, axis=0)
    return x[:,:,:,0][...,None]


def normalize(x):
    """Utility function to normalize a tensor by its L2 norm"""
    return (x + 1e-10) / (k.sqrt(k.mean(k.square(x))) + 1e-10)


##create data generator for test
test_path = "/home/vedantb/datav5/Test_v53_3c_by5.txt"
test_df = pd.read_csv(test_path,sep=".png ",header=None)
test_df[0] = test_df[0].astype(str)+'.png'
test_df[0]=test_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)
test_list_IDs = test_df[0].values
test_labels = test_df[1].values
test_gen = CustomDataGenerator.DataGenerator(test_list_IDs,test_labels,batch_size=BATCH_SIZE,n_channels=1,shuffle=False)

val_path = "/home/vedantb/datav5/Val_v53_3c_by5.txt"
val_df = pd.read_csv(val_path,sep=".png ",header=None)
val_df[0] = val_df[0].astype(str)+'.png'
val_df[0]=val_df[0].replace("/media/Newhdd2tb","/mnt/2tbhdd",regex=True)
val_list_IDs = val_df[0].values
val_labels = val_df[1].values

##define models dictionary
models = {}
models['resnet50_2']=ResNet50
# models['mobilenet'] = MobileNetV2
# models['alexnet']=AlexNet
# models['vgg16']=VGG16
# models['densenet'] = DenseNet169
# models['xception']=Xception


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
    for i in reversed(model.layers):
        if(len(i.output_shape)==4):
            layer_to_analyze=i.name
            break
    return(model,layer_to_analyze)


for name,m in models.items():
    print("name",name)
    model,layer_name = get_model(name)
    break
def run_shap(img_path,model,name):
    times_for_10_backgrounds = []
    times_for_shap_values = []
    shap_values = []
    shap_averaged = np.zeros((3,1,240,320,1))
    test_image = load_image(img_path)
    img_path = img_path.split('/')[-1:]
    count = 0
    for i in range(0,1000,10):
        count+=1
        print(count,i)
        background = np.zeros((10,240,320,1))
        for j in range(10):
            background[j,...]=load_image(val_list_IDs[i+j])
        t1 = time.time()
        ex = shap.DeepExplainer(model,background)
        t2 = time.time()
        shap_averaged += np.asarray(ex.shap_values(test_image)).reshape(3,1,240,320,1)
        t3 = time.time()
        times_for_10_backgrounds.append(t2-t1)
        times_for_shap_values.append(t3-t2)
        if(i%100==0):
            np.save('./shap_storage/{0}_sum_shap_values_{1}_backgr_image_{2}'.format(name,i,img_path),shap_averaged)

    records = {}

    records['num_values'] = len(times_for_shap_values)
    records['average_time_for_10_backgr'] = sum(times_for_10_backgrounds)/len(times_for_10_backgrounds)
    records['total_time_for_n_backgrounds'] = sum(times_for_10_backgrounds)
    records['average_shap_value_time'] = sum(times_for_shap_values)/len(times_for_shap_values)
    records['total_shap_values_time'] = sum(times_for_shap_values)
    np.save('./shap_storage/{0}_init_records_for_{1}'.format(name,img_path),records)

    shap_averaged /= count
    shap_list = []
    for i in range(3):
        shap_list.append(shap_averaged[0,...])
    t4 = time.time()
    shap.image_plot(shap_list,-test_image[0,...])
    t5 = time.time()
    records['image_plot_time_one_image'] = t5-t4
    np.save('./shap_storage/{0}_final_records_for_{1}'.format(name,img_path),records)
  
for name,m in models.items():
    print("name",name)
    model,layer_name = get_model(name)
    run_shap(test_list_IDs[100],model,name)

 

        
        