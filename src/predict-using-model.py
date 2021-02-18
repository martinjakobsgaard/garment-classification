# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

import numpy as np

import tensorflow.compat.v1 as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, color_mode="rgb", target_size=(100, 100))

	#img = load_img(filename, target_size=(640, 360))
	#res = cv2.resize(img, dsize=(54, 140), interpolation=cv2.INTER_CUBIC)
	# convert to array
	input_arr = img_to_array(img)
	input_arr = np.array([input_arr])
	# reshape into a single sample with 1 channel
	#img = img.reshape(100, 100, 3)
	# prepare pixel data
	input_arr = input_arr.astype('float32')
	input_arr = input_arr/255.0
	return input_arr

# load an image and predict the class
def run_example():
	# load the image
	input_arr = load_image('../images/validation/assorted-garments-blue/export_i0079_z1322.jpg')
	# load model
	model = load_model('final_model.h5')
	# predict the class
	result_prediction = model.predict(input_arr)
	print(np.argmax(result_prediction))

# entry point, run the example
run_example()
