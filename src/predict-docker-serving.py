# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import requests
import json

import numpy as np
import sys
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

to_res = (100, 100)

url = 'http://localhost:8501/v1/models/resnet-serving:predict'

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, color_mode="rgb", target_size=(100, 100))

    # img = load_img(filename, target_size=(640, 360))
    # res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
    # convert to array
    input_arr = img_to_array(img)
    input_arr = np.array([input_arr])
    # reshape into a single sample with 1 channel
    # img = img.reshape(100, 100, 3)
    # prepare pixel data
    input_arr = input_arr.astype('float32')
    input_arr = input_arr / 255.0
    return input_arr


def make_prediction(instances):
    data = json.dumps({"signature_name": "serving_default", "instances": instances.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions


# load an image and predict the class
def run_example():
    # load the image
    # input_arr = load_image('../images/validation/bed-linen-yellow/export_i0438_z1322.jpg')
    input_arr = load_image(str(sys.argv[1]))

    predictions = make_prediction(input_arr)

    labels = ["blue", "green", "striped", "yellow"]
    print(labels[np.argmax(predictions)])


# entry point, run the example
run_example()
