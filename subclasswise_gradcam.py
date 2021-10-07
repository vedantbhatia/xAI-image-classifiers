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
from tensorflow.keras.applications.densenet import DenseNet121
from collections import OrderedDict
from sklearn import metrics
from tensorflow.keras.preprocessing import image as Image
import cv2
##define macros
BATCH_SIZE = 64
GPU_ID = "2"
GPU_FRACTION = 0.65


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

##utility function to return the top 50 predictions per class, correct and wrong
# def get_top_predictions(y_true,y_pred_proba,n_classes=3,n_images):
#     size = y_pred_proba.shape[0]
#     y_true = y_true[:size]
#     y_pred = np.argmax(y_pred_proba,axis=1)
#     output = []
    
#     correct_indices = np.where(y_pred==y_true)[0]
#     for i in range(3):
#             class_indices = np.where(y_pred[correct_indices]==i)[0]
#             top_n_correct_indices = np.where(y_pred_proba[correct_indices[class_indices]][:,i].argsort(axis=0).argsort(axis=0)>=len(class_indices)-1)
#             final_top_correct = y_pred_proba[correct_indices[class_indices[top_n_correct_indices[0]]]]



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


##define models dictionary
models = {}
#models['resnet50_2']=ResNet50
# models['mobilenet'] = MobileNetV2
# models['alexnet']=AlexNet
#models['vgg16']=VGG16
models['densenet121'] = DenseNet121
# models['xception']=Xception



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

##helper function to run gradcam for a concat_df and save the images
def run_gradcam_df(df,model_name,model,sublclass):
    for i in range(df.shape[0]):
        path = df['path'].iloc[i]
        img = load_image(path)
        true_pred = df['label'].iloc[i]
        saved_pred = df['pred'].iloc[i]
        #change layer name if needed
        layer_name='conv5_block16_concat'
        class_idx,heatmap,heatmap_jet,superimposed_image,image_bbox = gradCAM(model,layer_name,img)
        plt.imsave("./best_models/{0}/top_results/superimposed_true_{1}_savedpred_{2}_pred{3}_sublclass_{4}_id_{5}.jpg".format(model_name,true_pred,saved_pred,class_idx,sublclass,i),superimposed_image)
        plt.imsave("./best_models/{0}/top_results/heatmapjet_true_{1}_savedpred_{2}_pred{3}_sublclass_{4}_id_{5}.jpg".format(model_name,true_pred,saved_pred,class_idx,sublclass,i),heatmap_jet)
        plt.imsave("./best_models/{0}/top_results/original_true_{1}_savedpred_{2}_pred{3}_sublclass_{4}_id_{5}.jpg".format(model_name,true_pred,saved_pred,class_idx,sublclass,i),img[0,:,:,0],cmap='gray')







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


##gradcam for one image
def gradCAM(model,layer_name,image):
    
    prob_predict = model.predict(image)
    #get class label
    # print(prob_predict,)
    class_idx = np.argmax(prob_predict[0])
    #get the last convolutional layer/the layer we want to analyze
    last_conv_layer = model.get_layer(layer_name)
    #get the gradients : loss wrt this layer
    grads = normalize(k.gradients(model.output[0,class_idx],last_conv_layer.output)[0])
    #pool the gradients channel wise: (x,y,z)=(z)
    pooled_grads = k.mean(grads, axis=(0,1,2))
    #get the value of this pooled gradient map and the output of our layer in question for this partiular input
    iterate = k.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value= iterate([image])
    #multiply each channel with the weights derived above i.e. pooled grads
    for i in range(conv_layer_output_value.shape[2]):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

    #generate heatmap
    heatmap = np.mean(conv_layer_output_value,axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap/=np.max(heatmap)
    heatmap = (cv2.resize(heatmap,(image.shape[2],image.shape[1]))*255.0).astype('uint8')
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#   heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    heatmap_jet = cv2.cvtColor(heatmap_jet,cv2.COLOR_BGR2RGB)

    #superimpose heatmap on image
    image_rgb = np.dstack((image[0,:,:,0],image[0,:,:,0],image[0,:,:,0]))
    image_scaled = ((np.maximum(image_rgb,0)/image_rgb.max()) *255.0).astype('uint8')
    superimposed_image = cv2.addWeighted(image_scaled,0.6,heatmap_jet,0.4,0)
    
    #hacky contour based bounding box generation
    image_bbox = np.copy(superimposed_image)
    heatmap = (heatmap>100)*heatmap
    _,contours,_ = cv2.findContours(heatmap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(image_bbox,(x,y),(x+w,y+h),(0,255,0),2)

    return(class_idx,heatmap,heatmap_jet,superimposed_image,image_bbox)

##gradcam++ for a single image
def gradCAMpp(model,layer_name,image):
    
    prob_predict = model.predict(image)
    #get class label
    class_idx = np.argmax(prob_predict[0])
    #get the last convolutional layer/the layer we want to analyze
    last_conv_layer = model.get_layer(layer_name)

    pred = model.output[0,class_idx]
    #get the gradients : loss wrt this layer
    # grads = normalize(k.gradients(model.output[0,class_idx],last_conv_layer.output)[0])
    grads = normalize(k.gradients(model.output[0,class_idx],last_conv_layer.output)[0])
    first_derivative = k.exp(pred)*grads
    second_derivative = k.exp(pred)*grads*grads
    third_derivative = k.exp(pred)*grads*grads*grads

    iterate = k.function([model.input],[pred,first_derivative,second_derivative,third_derivative,last_conv_layer.output,grads])
    pred,first_d,second_d,third_d,conv_output,grads= iterate([image])
    conv_sum_mapwise = np.sum(conv_output[0].reshape(-1,conv_output.shape[-1]),axis=0).reshape(1,1,conv_output.shape[-1])
    denom = (2.0*second_d[0]+conv_sum_mapwise*third_d[0])
    denom = np.where(denom != 0.0, denom, np.ones(denom.shape))
    alpha_k_c_i_j = (second_d[0])/denom

    #multiply each channel with the weights derived above i.e. pooled grads
    weights = np.max(first_d[0],0)
    alphas_thresholding = np.where(weights, alpha_k_c_i_j, 0.0)
    alpha_normalization_constant = np.sum(np.sum(alphas_thresholding, axis=0),axis=0)
    alpha_normalization_constant_processed = np.where(alpha_normalization_constant != 0.0, alpha_normalization_constant, np.ones(alpha_normalization_constant.shape))
    alpha_k_c_i_j /= alpha_normalization_constant_processed.reshape((1,1,conv_output.shape[-1]))


    deep_linearization_weights = np.sum((weights*alpha_k_c_i_j).reshape((-1,first_d[0].shape[-1])),axis=0)
    #print deep_linearization_weights
    grad_CAM_map = np.sum(deep_linearization_weights*conv_output, axis=-1)
    heatmap = np.maximum(grad_CAM_map, 0)
    heatmap /= np.max(heatmap) # scale 0 to 1.0   

    #generate heatmap
    heatmap = (cv2.resize(heatmap[0],(image.shape[2],image.shape[1]))*255.0).astype('uint8')
    heatmap_jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
#   heatmap = cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)
    heatmap_jet = cv2.cvtColor(heatmap_jet,cv2.COLOR_BGR2RGB)

    #superimpose heatmap on image
    image_rgb = np.dstack((image[0,:,:,0],image[0,:,:,0],image[0,:,:,0]))
    image_scaled = ((np.maximum(image_rgb,0)/image_rgb.max()) *255.0).astype('uint8')
    superimposed_image = cv2.addWeighted(image_scaled,0.6,heatmap_jet,0.4,0)
    
    #hacky contour based bounding box generation
    image_bbox = np.copy(superimposed_image)
    heatmap = (heatmap>100)*heatmap
    _,contours,_ = cv2.findContours(heatmap,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        cv2.rectangle(image_bbox,(x,y),(x+w,y+h),(0,255,0),2)

    return(class_idx,heatmap,heatmap_jet,superimposed_image,image_bbox)

##run code and save files
for name,m in models.items():
    print("name",name)
    model,layer_name = get_model(name)
    y_pred = np.load('./best_models/{0}/pred_test.npy'.format(name),allow_pickle=True)
    concat_df = build_df(test_list_IDs,test_labels,y_pred)
    for subclass in subclass_names:
        print("sublclass",subclass)
        subclass_df = get_subclass_dataset(concat_df,subclass)
        top_true_true,top_true_false = get_top_n(subclass_df,2,0,5)
        top_false_false, top_false_true = get_top_n(subclass_df,0,2,5)
        for i in [top_true_true,top_true_false,top_false_true,top_false_false]:
            run_gradcam_df(i,name,model,subclass)

    


