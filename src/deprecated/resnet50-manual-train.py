import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from sklearn.model_selection import train_test_split 

from keras.preprocessing.image import ImageDataGenerator

import tensorflow.compat.v1 as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# create generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('../images/dataset-1/train/',  color_mode='rgb', target_size=(100, 100), class_mode='sparse', batch_size=1200)
test_it = datagen.flow_from_directory('../images/dataset-1/test/', color_mode='rgb', target_size=(100, 100), class_mode='sparse', batch_size=400)

train_im, train_lab = train_it.next()
test_im, test_lab = test_it.next()

# Normalize the images to pixel values (0, 1)
train_im, test_im = train_im/255.0 , test_im/255.0

# Check the format of the data 
print ("train_im, train_lab types: ", type(train_im), type(train_lab))

# check the shape of the data
print ("shape of images and labels array: ", train_im.shape, train_lab.shape) 
print ("shape of images and labels array ; test: ", test_im.shape, test_lab.shape)

# define class types
class_types = ['assorted-garments-blue', 'assorted-garments-green', 'bed-linen-striped', 'bed-linen-yellow']

# Convert to categoricals -_-
print("Converting to categoricals...")
train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=4, dtype='uint8')
test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=4, dtype='uint8')


# Split datasets
print("Splitting datasets...")
train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.20, 
                                                            stratify=train_lab_categorical, 
                                                            random_state=40, shuffle = True)
# define batch size
batch_size = 64 # try several values

train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.1, 
                                                                width_shift_range=0.01, 
                                                                height_shift_range = 0.01, 
                                                                horizontal_flip=False)
 
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size) # train_lab is categorical 
valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size) # so as valid_lab

def res_identity(x, filters): 
    ''' renet block where dimension doesnot change.
        The skip connection is just simple identity conncection
        we will have 3 blocks and then input will be added
        '''
    x_skip = x # this will be used for addition with the residual block 
    f1, f2 = filters

    #first block 
    x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    #second block # bottleneck (but size kept same with padding)
    x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Activation(activations.relu)(x)

    # third block activation used after adding the input
    x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    # x = Activation(activations.relu)(x)

    # add the input 
    x = Add()([x, x_skip])
    x = Activation(activations.relu)(x)

    return x

def res_conv(x, s, filters):
	'''
	here the input size changes, when it goes via conv blocks
	so the skip connection uses a projection (conv layer) matrix
	''' 
	x_skip = x
	f1, f2 = filters

	# first block
	x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
	# when s = 2 then it is like downsizing the feature map
	x = BatchNormalization()(x)
	x = Activation(activations.relu)(x)

	# second block
	x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
	x = BatchNormalization()(x)
	x = Activation(activations.relu)(x)

	#third block
	x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
	x = BatchNormalization()(x)

	# shortcut 
	x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
	x_skip = BatchNormalization()(x_skip)

	# add 
	x = Add()([x, x_skip])
	x = Activation(activations.relu)(x)

	return x

### Combine the above functions to build 50 layers resnet. 
def resnet50():
	input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3])) # cifar 10 images size
	x = ZeroPadding2D(padding=(3, 3))(input_im)

	# 1st stage
	# here we perform maxpooling, see the figure above

	x = Conv2D(64, kernel_size=(7, 7), strides=(2, 2))(x)
	x = BatchNormalization()(x)
	x = Activation(activations.relu)(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	#2nd stage 
	# frm here on only conv block and identity block, no pooling

	x = res_conv(x, s=1, filters=(64, 256))
	x = res_identity(x, filters=(64, 256))
	x = res_identity(x, filters=(64, 256))

	# 3rd stage

	x = res_conv(x, s=2, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))
	x = res_identity(x, filters=(128, 512))

	# 4th stage

	x = res_conv(x, s=2, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))
	x = res_identity(x, filters=(256, 1024))

	# 5th stage

	x = res_conv(x, s=2, filters=(512, 2048))
	x = res_identity(x, filters=(512, 2048))
	x = res_identity(x, filters=(512, 2048))

	# ends with average pooling and dense connection

	x = AveragePooling2D((2, 2), padding='same')(x)

	x = Flatten()(x)
	x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class

	# define the model 

	model = Model(inputs=input_im, outputs=x, name='Resnet50')

	return model

### Define some Callbacks
def lrdecay(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr
    # if epoch < 40:
    #   return 0.01
    # else:
    #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay  

def earlystop(mode):
	if mode=='acc':
		estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
	elif mode=='loss':
		estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
	return estop

resnet50_model = resnet50()
#resnet50_model.summary()

resnet50_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), 
                       metrics=['acc'])

batch_size=batch_size
print(batch_size)

resnet_train = resnet50_model.fit(train_set_conv, 
                                  epochs=160, 
                                  steps_per_epoch=train_im.shape[0]/batch_size, 
                                  validation_steps=valid_im.shape[0]/batch_size, 
                                  validation_data=valid_set_conv,
								  callbacks= [lrdecay])

resnet50_model.save('../models/resnet50_manual_train.h5')

