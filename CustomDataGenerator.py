import numpy as np
import tensorflow.keras as keras
import cv2 as cv

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=16, dim=(240,320), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        labels_temp = [self.labels[k] for k in indexes]
        # Generate data
        X, y = self.__data_generation(list_IDs_temp,labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp,labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
#             data = self.list_IDs[i]
            img = cv.imread(ID,0 if self.n_channels==1 else 1)
            img = cv.resize(img,(320,240),0)
#             print(np.max(img),img.shape,img.dtype)
            img = img/255.0
#             print(np.max(img),img.shape,img.dtype)

#             X[i,] = np.load('data/' + ID + '.npy')
            X[i,]=img[...,None]
            # Store class
#             y[i] = self.labels[ID]
#         print(X,keras.utils.to_categorical(labels_temp, num_classes=self.n_classes))
        return X, keras.utils.to_categorical(labels_temp, num_classes=self.n_classes)