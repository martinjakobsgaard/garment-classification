# Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import kerastuner as kt

# Allow GPU growth (avoid memory problems)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def build_model(hp):
    model = K.Sequential()
    model.add(K.layers.Flatten(input_shape=(200, 200,3)))

    for i in range(hp.Int('num_layers', 2, 15)): # was 2, 15
        model.add(K.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
    model.add(K.layers.Dense(5, activation='softmax'))
    model.compile(
        optimizer=K.optimizers.Adam(
            hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    return model

# Fetch data
# Note: This is a hacky way of getting all images/labels. The iterator is only used once, to get all data.
#       The batch size is the size of the dataset, and _does not_ match the later one.

### IMPORT DATA #####################################################
print("\n=== IMPORT DATA =======================\n")
data_generator_train = ImageDataGenerator()
iterator_train = data_generator_train.flow_from_directory(
    '../images/garment-dataset-2/train/',
    color_mode='rgb',
    target_size=(200, 200),
    class_mode='sparse',
    batch_size=8743)
print("Train data batch size: ", iterator_train.batch_size)
img_train, label_train = iterator_train.next()

data_generator_test= ImageDataGenerator()
iterator_test = data_generator_test.flow_from_directory(
    '../images/garment-dataset-2/test/',
    color_mode='rgb',
    target_size=(200, 200),
    class_mode='sparse',
    batch_size=1494)
print("Test data batch size: ", iterator_test.batch_size)
img_test, label_test = iterator_test.next()

# Normalize the images to pixel values (0, 1)
img_train = img_train.astype('float32')/255.0
img_test = img_test.astype('float32')/255.0

### DO AI ##########################################################
#src: https://github.com/keras-team/keras-tuner
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=15,
                     factor=3,
                     directory='hypermodel-2',
                     project_name='intro_to_kt')

#tuner = kt.RandomSearch(build_model,
#                     objective='val_accuracy',
#                     max_trials=10,
#                     directory='my_dir',
#                     project_name='intro_to_kt')

stop_early = K.callbacks.EarlyStopping(monitor='val_loss', patience=5)
tuner.search(img_train, label_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
#tuner.search(img_train, label_train, epochs=50, validation_split=0.2)


# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.get_best_models(num_models=1)[0]

print("TRAIN THE MODEL")
#Find the optimal number of epochs to train the model with the hyperparameters obtained from the search.
# Build the model with the optimal hyperparameters and train it on the data for 50 epochs
#model = tuner.hypermodel.build(best_hps)
#history = model.fit(img_train, label_train, epochs=500, validation_split=0.2)
history = best_model.fit(img_train, label_train, epochs=200, validation_split=0.2)


print("EVALUATE MODEL")
eval_result = best_model.evaluate(img_test, label_test)
print("[test loss, test accuracy]:", eval_result)

