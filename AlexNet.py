"""
AlexNet Keras Implementation

BibTeX Citation:

@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

def alexnet_model(img_shape=(240, 320, 1), n_classes=3, l2_reg=0.,
	weights=None):

	# Initialize model
# 	alexnet = Sequential()

	model = Sequential()

	# 1st Convolutional Layer
	model.add(Conv2D(filters=96, input_shape=img_shape, kernel_size=(11,11), strides=(4,4), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# 2nd Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# 3rd Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))

	# 4th Convolutional Layer
	model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))

	# 5th Convolutional Layer
	model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
	model.add(Activation('relu'))
	# Max Pooling
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

	# Passing it to a Fully Connected layer
	model.add(Flatten())
	# 1st Fully Connected Layer
	model.add(Dense(4096, input_shape=(224*224*3,)))
	model.add(Activation('relu'))
	# Add Dropout to prevent overfitting
	model.add(Dropout(0.4))

	# 2nd Fully Connected Layer
	model.add(Dense(4096))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))

	# 3rd Fully Connected Layer
	model.add(Dense(1000))
	model.add(Activation('relu'))
	# Add Dropout
	model.add(Dropout(0.4))

	# Output Layer
	model.add(Dense(n_classes))
	model.add(Activation('softmax'))


	if weights is not None:
		model.load_weights(weights)

	return model

def parse_args():
	"""
	Parse command line arguments.

	Parameters:
		None
	Returns:
		parser arguments
	"""
	parser = argparse.ArgumentParser(description='AlexNet model')
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	optional.add_argument('--print_model',
		dest='print_model',
		help='Print AlexNet model',
		action='store_true')
	parser._action_groups.append(optional)
	return parser.parse_args()

if __name__ == "__main__":
	# Command line parameters
	args = parse_args()

	# Create AlexNet model
	model = alexnet_model()

	# Print
	if args.print_model:
		model.summary()