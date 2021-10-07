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
import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import gradcampp
import gradutils as util
import imutils
##define macros
BATCH_SIZE = 64
GPU_ID = "1"
GPU_FRACTION = 0.5


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

##utility function to get 10 correct and 10 incorrect predictions per class
def get_random_images(y_true,y_pred,n_classes=3,n_images=10):
    y_true = y_true[:y_pred.shape[0]]
    y_pred = np.argmax(y_pred,axis=1)
    image_ids = {}
    correct = np.nonzero(y_true==y_pred)[0]
    incorrect = np.nonzero(y_true!=y_pred)[0]
    for i in range(n_classes):
        image_ids[i]=[]
        # print([y_true[correct]==i])
        indices_correct = np.random.permutation(np.nonzero(y_true[correct]==i)[0])[:n_images]
        indices_incorrect = np.random.permutation(np.nonzero(y_true[incorrect]==i)[0])[:n_images]
        # print("for class",i,y_true[indices_correct])
        image_ids[i].append(correct[indices_correct])
        image_ids[i].append(incorrect[indices_incorrect])
    return(image_ids)


##utility function to load an image in the correct format for visualization
def load_image(path, preprocess=True):
    """Load and preprocess image."""
    x = Image.load_img(path, target_size=(240, 320))
    if preprocess:
        x = Image.img_to_array(x)
#         x = x
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





##define the different sub-classes of images possible
subclass_names = ["Abdomen","Cardiac","Early_OB","Gyn","NeoHead","NON_OB","NON_TISSUE","OB","OB_T1","OB_T2","OB_T3","Vascular"]

##utility function to build a df in the form required by get_top_n
def build_df(paths,labels,predictions,n_classes=3):
    df = pd.DataFrame()
    df['path']=paths
    df['label']=labels
    df = df.head(predictions.shape[0])
    df['pred'] = np.argmax(predictions,axis=1)
    for i in range(n_classes):
        column_name = 'proba_{0}'.format(i)
        df[column_name] = predictions[:,i]
    return(df)


##utility function to split the classwise dataset into subclasswise dataset
def get_subclass_dataset(df,subclass):
    subclass_df = df.loc[df['path'].str.contains(subclass)]
    return(subclass_df)

##utility fucntion to get the top n correct and wrong predictions of a particular class - concat_df: path, label, prediction, [predict_probas]
def get_top_n(df,class_id,false_id,n):
    true_preds = df.loc[(df['pred']==class_id) & (df['label']==class_id)]
    wrong_preds = df.loc[(df['pred']==false_id) & (df['label']==class_id)]

    top_true = true_preds.nlargest(n,'proba_{0}'.format(class_id))
    top_false = wrong_preds.nlargest(n,'proba_{0}'.format(false_id))

    return(top_true,top_false)



##helper function to run GradCAM for a concat_df and save the images
def run_gradcam_df(df,model_dict,model_name,model,sublclass):
    for i in range(df.shape[0]):
        path = df['path'].iloc[i]
        img = load_image(path)
        img = img[:,None,:,:,0]
        img_3 = np.dstack((img[0,0,...],img[0,0,...],img[0,0,...])).astype('uint8')  
#         print(img_3.shape,print()
        true_pred = df['label'].iloc[i]
        saved_pred = df['pred'].iloc[i]
        for layer_name in layer_names:
            model_dict['layer_name']=layer_name
            grad=gradcampp.GradCAM(model_dict,channels=1)
            inp = Variable(torch.FloatTensor(img))
#             print(inp.shape,img.shape)
#             print(inp.shape)
            mask_gc, logit=grad(inp)
#             print("mask",mask_gc.shape)
            heatmap_gc=cv.applyColorMap(np.uint8(255 * mask_gc[0,0,...]), cv.COLORMAP_JET)
            heatmap_gc = cv.cvtColor(heatmap_gc,cv.COLOR_BGR2RGB)
            cam_result_gc=cv.addWeighted(heatmap_gc,0.4,img_3,0.6,0)
            plt.imsave("./best_models/{0}/top_results/mask_gc_true_{1}_pred_{2}_sublclass_{3}_id_{4}_layer_{5}.jpg".format(model_name,true_pred,saved_pred,sublclass,i,layer_name),mask_gc[0,0,:,:],cmap='gray')
            plt.imsave("./best_models/{0}/top_results/superimposed_true_{1}_pred_{2}_sublclass_{3}_id_{4}_layer_{5}.jpg".format(model_name,true_pred,saved_pred,sublclass,i,layer_name),cam_result_gc)
            plt.imsave("./best_models/{0}/top_results/original_true_{1}_pred_{2}_sublclass_{3}_id_{4}_layer_{5}.jpg".format(model_name,true_pred,saved_pred,sublclass,i,layer_name),img[0,0,:,:],cmap='gray')
            plt.imsave("./best_models/{0}/top_results/heatmap_true_{1}_pred_{2}_sublclass_{3}_id_{4}_layer_{5}.jpg".format(model_name,true_pred,saved_pred,sublclass,i,layer_name),heatmap_gc)


import torch,imp
from torchvision import transforms
from torch.autograd import Variable
class pytorch_model():
    def __init__(self):
        MainModel = imp.load_source('MainModel', "../XAI/utils/pytorch_googlenet.py")
        the_model = torch.load("../XAI/utils/portal/pytorch_googlenet.model")
        the_model.eval()
        self.model = the_model

    def predict_proba(self,arr):
        self.model.eval()
        data = torch.from_numpy(arr)
        data = torch.autograd.Variable(data, requires_grad = False).float()
        predict = self.model(data).detach().numpy()
        return(predict)
    def predict(self,arr):
        return(np.argmax(self.predict_proba(arr),axis=1))




##run code and save files
layer_names = ['inception_5b_1x1','inception_5b_3x3', 'inception_5b_5x5']
model = pytorch_model()
name = "googlenet"
model_dict = dict(type='googlenet', arch=model.model, input_size=(240, 320))
print("name",name)
y_pred = np.load('./best_models/{0}/pred_test.npy'.format(name),allow_pickle=True)
concat_df = build_df(test_list_IDs,test_labels,y_pred)
for subclass in subclass_names:
    print("sublclass",subclass)
    subclass_df = get_subclass_dataset(concat_df,subclass)
    top_true_true,top_true_false = get_top_n(subclass_df,2,0,5)
    top_false_false, top_false_true = get_top_n(subclass_df,0,2,5)
    for i in [top_true_true,top_true_false,top_false_true,top_false_false]:
        run_gradcam_df(i,model_dict,name,model,subclass)


