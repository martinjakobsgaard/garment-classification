# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def conf_matrix(predictions):
    ''' Plots conf. matrix and classification report '''
    cm=confusion_matrix(test_lab, np.argmax(np.round(predictions), axis=1))
    print("Classification Report:\n")
    cr=classification_report(test_lab,
                                np.argmax(np.round(predictions), axis=1),
                                target_names=[class_types[i] for i in range(len(class_types))])
    print(cr)
    plt.figure(figsize=(12,12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [class_types[i] for i in range(len(class_types))],
                yticklabels = [class_types[i] for i in range(len(class_types))], fmt="d")
    fig = sns_hmp.get_figure()


# create generator
datagen = ImageDataGenerator()

to_res = (125, 125)
batch_size = 64

test_it = datagen.flow_from_directory('../images/garment-dataset-towel-linen/test/', color_mode='rgb', target_size=(125, 125), class_mode='sparse', batch_size=389)
test_im, test_lab = test_it.next()

test_im = test_im/255.0

class_types = ['assorted-garments-blue', 'assorted-garments-green']

# check the shape of the data
print("shape of images and labels array: ", test_im.shape, test_lab.shape)

test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=2, dtype='uint8')

model = load_model('../models/resnet50.h5')

pred_class_resnet50 = model.predict(test_im)

conf_matrix(pred_class_resnet50)