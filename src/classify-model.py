# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np
import sys
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

to_res = (224, 224)


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="rgb", target_size=(224, 224))

    # img = load_img(filename, target_size=(640, 360))
    # res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    # convert to array
    input_arr = img_to_array(img)
    input_arr = np.array([input_arr])
    # reshape into a single sample with 1 channel
    # img = img.reshape(100, 100, 3)
    # prepare pixel data
    input_arr = input_arr.astype('float32')
    #input_arr = input_arr / 255.0
    # train_im = train_im/255.0
    # train_im = tf.keras.applications.resnet50.preprocess_input(train_im)
    input_arr = tf.keras.applications.resnet50.preprocess_input(input_arr)
    return input_arr


# load an image and predict the class
def run_example():
    
    # load the image
    #input_arr = load_image('../images/validation/bed-linen-yellow/export_i0438_z1322.jpg')
    input_arr = load_image(str(sys.argv[1]))

    # load model
    model = load_model('../models/resnet50.h5')
    # predict the class
    result_prediction = model.predict(input_arr)
    #labels = ["blue", "green", "striped", "yellow"]
    labels = ['bed-linen-white', 'towel-white']
    print("Result: ", result_prediction)
    print(labels[np.argmax(result_prediction)])


# entry point, run the example
run_example()
