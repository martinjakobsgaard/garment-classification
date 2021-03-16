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

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Gpu warning fix
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def conf_matrix(predictions, labels):
    cm=confusion_matrix(labels, np.argmax(np.round(predictions), axis=1))
    print("Classification Report:\n")
    cr=classification_report(test_lab,
                                np.argmax(np.round(predictions), axis=1),
                                target_names=[setting_class_types[i] for i in range(len(setting_class_types))])
    print(cr)
    plt.figure(figsize=(12, 12))
    sns_hmp = sns.heatmap(cm, annot=True, xticklabels = [setting_class_types[i] for i in range(len(setting_class_types))],
                yticklabels = [setting_class_types[i] for i in range(len(setting_class_types))], fmt="d")
    fig = sns_hmp.get_figure()


# Settings
setting_resolution = (224, 224)
setting_batch_size = 1201
setting_class_count = 4
setting_class_types = ['bed-linen-yellow-stripe', 'blue-pants', 'green-pants', 'towel-white']

# Load model
to_res = setting_resolution
model = load_model('../models/RESNET50-NO-LINEN_res224_batch64_epoch50_frozen0--1_class4_size7029.h5')

# Fetch data
datagen = ImageDataGenerator()
test_it = datagen.flow_from_directory('../images/garment-dataset-2-no-linen/test/', color_mode='rgb', target_size=setting_resolution, class_mode='sparse', batch_size=setting_batch_size)
test_im, test_lab = test_it.next()
test_im = tf.keras.applications.resnet50.preprocess_input(test_im)
test_lab_categorical = tf.keras.utils.to_categorical(test_lab, num_classes=setting_class_count, dtype='uint8')

# Get predictions
predictions = [[0, 0, 0, 0]]
for im in test_im:
    format_test_im = im[None, :, :, :]
    prediction = model.predict(format_test_im)
    predictions = np.append(predictions, prediction, axis=0)
predictions = np.delete(predictions, 0, 0)

# Print results
conf_matrix(predictions, test_lab)
