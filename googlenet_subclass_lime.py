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


##wrapper function for LIME predict - convert 3 channel request to single channel request
def predict_facade(imgs):
    global PREDICT_FACADE_MODEL
    grayscale_imgs = np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2],1))
    grayscale_imgs = np.mean(imgs,axis=-1)
    # pred = PREDICT_FACADE_MODEL.predict(grayscale_imgs[...,None])
    im = grayscale_imgs.transpose((0,2,1))[:,None,...]
    pred = PREDICT_FACADE_MODEL.predict_proba(im)
    return(pred)


##helper function to run LIME for a concat_df and save the images
def run_lime_df(df,explainer,model_name,model,sublclass):
    for i in range(df.shape[0]):
        path = df['path'].iloc[i]
        img = load_image(path)
        img_3 = np.dstack((img[0,...],img[0,...],img[0,...]))
#         print(img_3.shape)
        true_pred = df['label'].iloc[i]
        saved_pred = df['pred'].iloc[i]
        explanation = explainer.explain_instance(img_3, predict_facade, top_labels=3, hide_color=0, num_samples=1000,labels=[0,1,2])
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=6, hide_rest=False)
        top_6_all = mark_boundaries(temp / 2 + 0.5, mask)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=3, hide_rest=False)
        top_3_positive = mark_boundaries(temp / 2 + 0.5, mask)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=2, hide_rest=False)
        top_2_positive = mark_boundaries(temp / 2 + 0.5, mask)
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=1, hide_rest=False)
        top_1_positive = mark_boundaries(temp / 2 + 0.5, mask)
        np.save("./best_models/{0}/lime_results/explanation_true_{1}_pred_{2}_sublclass_{3}_id_{4}.npy".format(model_name,true_pred,saved_pred,sublclass,i),explanation)
        plt.imsave("./best_models/{0}/lime_results/top_6_all_true_{1}_pred_{2}_sublclass_{3}_id_{4}.jpg".format(model_name,true_pred,saved_pred,sublclass,i),top_6_all)
        plt.imsave("./best_models/{0}/lime_results/top_3_positive_true_{1}_pred_{2}_sublclass_{3}_id_{4}.jpg".format(model_name,true_pred,saved_pred,sublclass,i),top_3_positive)
        plt.imsave("./best_models/{0}/lime_results/top_2_positive_true_{1}_pred_{2}_sublclass_{3}_id_{4}.jpg".format(model_name,true_pred,saved_pred,sublclass,i),top_2_positive)
        plt.imsave("./best_models/{0}/lime_results/top_1_positive_true_{1}_pred_{2}_sublclass_{3}_id_{4}.jpg".format(model_name,true_pred,saved_pred,sublclass,i),top_1_positive)
        plt.imsave("./best_models/{0}/lime_results/original_true_{1}_pred_{2}_sublclass_{3}_id_{4}.jpg".format(model_name,true_pred,saved_pred,sublclass,i),img[0,:,:,0],cmap='gray')



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

##run code and save files
model = pytorch_model()
name = "googlenet"
explainer = lime_image.LimeImageExplainer()
print("name",name)
PREDICT_FACADE_MODEL = model
y_pred = np.load('./best_models/{0}/pred_test.npy'.format(name),allow_pickle=True)
concat_df = build_df(test_list_IDs,test_labels,y_pred)
for subclass in subclass_names:
    print("sublclass",subclass)
    subclass_df = get_subclass_dataset(concat_df,subclass)
    top_true_true,top_true_false = get_top_n(subclass_df,2,0,5)
    top_false_false, top_false_true = get_top_n(subclass_df,0,2,5)
    for i in [top_true_true,top_true_false,top_false_true,top_false_false]:
        run_lime_df(i,explainer,name,model,subclass)


