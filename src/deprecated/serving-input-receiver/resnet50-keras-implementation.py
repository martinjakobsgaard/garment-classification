#Imports
#import matplotlib.pyplot as plt
#import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sip import ServingInputReceiver


# Gpu warning fix
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


# create generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_it = datagen.flow_from_directory('../../images/filtered-dataset/train/',  color_mode='rgb', target_size=(100, 100), class_mode='sparse', batch_size=1200)

train_im, train_lab = train_it.next()

# Normalize the images to pixel values (0, 1)
train_im = train_im/255.0

# Check the format of the data 
print("train_im, train_lab types: ", type(train_im), type(train_lab))

# check the shape of the data
print("shape of images and labels array: ", train_im.shape, train_lab.shape)

# define class types
class_types = ['assorted-garments-blue', 'assorted-garments-green', 'bed-linen-striped', 'bed-linen-yellow']

# Convert to categoricals -_-
print("Converting to categoricals...")
train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=4, dtype='uint8')



# Split datasets
print("Splitting datasets...")
train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.20, 
                                                            stratify=train_lab_categorical, 
                                                            random_state=40, shuffle = True)
# define batch size
batch_size = 64 # try several values

train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.1, 
                                                                width_shift_range=0.01, 
                                                                height_shift_range=0.01,
                                                                horizontal_flip=False)
 
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size) # train_lab is categorical 
valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size) # so as valid_lab

# Defining model
input_shape = K.Input(shape=(100, 100, 3))
res_model = K.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_shape)

# If freeze layers in model
#for layer in res_model.layers:
#    layer.trainable = False

to_res = (100, 100)

model = K.models.Sequential()
model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
model.add(res_model)
model.add(K.layers.Flatten())
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(256, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(128, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(64, activation='relu'))
model.add(K.layers.Dropout(0.5))
model.add(K.layers.BatchNormalization())
model.add(K.layers.Dense(4, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath="../../models/ModelCheckPointKerasImpleResNet50.h5",
                                          monitor="val_acc", mode="max", save_best_only=True)

model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
resnet_train = model.fit(train_set_conv, epochs=10, verbose=1,
                    validation_data=valid_set_conv, callbacks=[check_point])


model.save('../resnet-input-b64-2/mymodel.h5', save_format='h5')

