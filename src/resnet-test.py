# Garment classification python

# Keras import

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

import tensorflow.compat.v1 as tf

from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# create generator
datagen = ImageDataGenerator()
# load and iterate training dataset
train_it = datagen.flow_from_directory('../images/dataset-1/train/',  color_mode='rgb', target_size=(100, 100), class_mode='categorical', batch_size=16)
# load and iterate validation dataset
val_it = datagen.flow_from_directory('../images/dataset-1/validation/', color_mode='rgb', target_size=(100, 100), class_mode='categorical', batch_size=16)
# load and iterate test dataset
test_it = datagen.flow_from_directory('../images/dataset-1/test/', color_mode='rgb', target_size=(100, 100), class_mode='categorical', batch_size=16)

batch_size = 64 # try several values

train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.1, 
                                                                width_shift_range=0.01, 
                                                                height_shift_range = 0.01, 
                                                                horizontal_flip=False)
 
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_im, train_lab = train_it.next()
valid_im, valid_lab = val_it.next()

train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size) # train_lab is categorical 
valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size) # so as valid_lab

class_types = ['assorted-garments-blue', 'assorted-garments-green', 'bed-linen-striped', 'bed-linen-yellow']


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

# define cnn model
def resnet50():
	input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3]))
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

	# compile model
	model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), 
                       metrics=['acc'])

	resnet_train = model.fit(train_set_conv, 
                                  epochs=160, 
                                  steps_per_epoch=train_im.shape[0]/batch_size, 
                                  validation_steps=valid_im.shape[0]/batch_size, 
                                  validation_data=valid_set_conv)

	return model

# run the test harness for evaluating a model
def run_test_harness():

	# define model
	model = resnet50()
	# fit model
	model.fit_generator(train_it, steps_per_epoch=64, validation_data=val_it, validation_steps=8)
	# evaluate model
	loss = model.evaluate_generator(test_it, steps=16)
	# save model
	model.save('final_model.h5')

# entry point, run the test harness
run_test_harness()
