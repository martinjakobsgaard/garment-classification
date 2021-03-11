# Imports
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Allow GPU growth (avoid memory problems)
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# Settings
setting_resolution = 224
setting_batch_size = 64
setting_epochs = 100
setting_frozen = True # <- was true
setting_frozen_min = 0
setting_frozen_max = -1 # max
setting_class_count = 5
setting_dataset_size = 8743
setting_path = '../images/garment-dataset-2/train/'

# Generate export name unique to training
export_suffix = "resnet152"
export_suffix += ("_res" + str(setting_resolution))
export_suffix += ("_batch" + str(setting_batch_size))
export_suffix += ("_epoch" + str(setting_epochs))
export_suffix += ("_frozen" + str(setting_frozen))
if setting_frozen:
    export_suffix += ("index" + str(setting_frozen_min) + "-" + str(setting_frozen_max))
export_suffix += ("_class" + str(setting_class_count))
export_suffix += ("_size" + str(setting_dataset_size))

# Fetch data
# Note: This is a hacky way of getting all images/labels. The iterator is only used once, to get all data.
#       The batch size is the size of the dataset, and _does not_ match the later one.
print("\n=== IMPORT DATA =======================\n")
datagen = ImageDataGenerator()
train_it = datagen.flow_from_directory(
    setting_path,
    color_mode='rgb',
    target_size=(setting_resolution, setting_resolution),
    class_mode='sparse',
    batch_size=setting_dataset_size)
print("Train data batch size: ", train_it.batch_size)
train_im, train_lab = train_it.next()

# Normalize the images to pixel values (0, 1)
#train_im = train_im/255.0
train_im = tf.keras.applications.resnet.preprocess_input(train_im)
    #resnet152.preprocess_input(train_im)

# Inspect imported data
print("train_im type:\t\t", type(train_im))
print("train_lab type:\t\t", type(train_lab))
print("train_im shape:\t\t", train_im.shape)
print("train_lab shape:\t", train_im.shape)

# define class types
#class_types = ['assorted-garments-blue', 'assorted-garments-green', 'bed-linen-striped', 'bed-linen-yellow']

# Convert to categoricals
print("Converting to categoricals...")
train_lab_categorical = tf.keras.utils.to_categorical(train_lab, num_classes=setting_class_count, dtype='uint8')

# Split datasets
print("Splitting datasets...")
train_im, valid_im, train_lab, valid_lab = train_test_split(train_im,
                                                            train_lab_categorical,
                                                            test_size=0.20,
                                                            stratify=train_lab_categorical, 
                                                            random_state=40, shuffle=True)

train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.1, 
                                                                width_shift_range=0.01, 
                                                                height_shift_range=0.01,
                                                                horizontal_flip=False)
 
valid_DataGen = tf.keras.preprocessing.image.ImageDataGenerator()

train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=setting_batch_size) # train_lab is categorical
valid_set_conv = valid_DataGen.flow(valid_im, valid_lab, batch_size=setting_batch_size) # so is valid_lab

# Defining model
print("\n=== DEFINE MODEL ======================\n")
print("Defining input shape...")
input_shape = K.Input(shape=(setting_resolution, setting_resolution, 3))
print("Shape: ", input_shape)
print("Defining ResNet1052...")

# Source: https://keras.io/api/applications/resnet/#resnet152-function
#res_model = K.applications.ResNet50(include_top=True, weights="imagenet")
res_model = K.applications.ResNet152(include_top=False, weights="imagenet", input_tensor=input_shape, pooling=max, classes=setting_class_count) # <- Good input
#res_model = K.applications.ResNet50(include_top=True, weights=None, input_tensor=input_shape, pooling=max, classes=setting_class_count) # <- Good input

# res_model = K.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_shape, pooling=max, classes=setting_class_count) # <- Good input

# If freeze layers in model
if setting_frozen:
    if setting_frozen_min == -1:
        for layer in res_model.layers[:setting_frozen_max]:
            layer.trainable = False
    elif setting_frozen_max == -1:
        for layer in res_model.layers[setting_frozen_min:]:
            layer.trainable = False
    else:
        for layer in res_model.layers[setting_frozen_min:setting_frozen_max]:
            layer.trainable = False

print("Defining classification layer...")
to_res = (setting_resolution, setting_resolution)
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
model.add(K.layers.Dense(setting_class_count, activation='softmax'))

check_point = K.callbacks.ModelCheckpoint(filepath="../models/resnet152" + export_suffix + "-checkpoint.h5",
                                          monitor="val_accuracy", mode="max", save_best_only=True)

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])
response = input("Do you want to start training? (Y/n) ")
if response == "n":
    exit(0)

# Train model
print("\n=== TRAIN MODEL =======================\n")
resnet_train = model.fit(train_set_conv, batch_size=setting_batch_size, epochs=setting_epochs, verbose=1,
                    validation_data=valid_set_conv, callbacks=[check_point])
model.summary()
val_acc_per_epoch = resnet_train.history['val_accuracy']
val_acc_best = max(val_acc_per_epoch)
print("Best val acc:", val_acc_best)

# Export model
model_export_name = "../models/resnet152" + export_suffix + ".h5"
model.save(model_export_name, save_format='h5')

# Optional: Plot train and validation curves
verbose = True
if verbose:
    print("\n=== PLOT MODEL ========================\n")
    loss = resnet_train.history['loss']
    v_loss = resnet_train.history['val_loss']
    acc = resnet_train.history['accuracy']
    v_acc = resnet_train.history['val_accuracy']
    epochs = range(len(loss))
    fig = plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.yscale('log')
    plt.plot(epochs, loss, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Loss')
    plt.plot(epochs, v_loss, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Loss')
    plt.ylim(0.3, 100)
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, linestyle='--', linewidth=3, color='orange', alpha=0.7, label='Train Acc')
    plt.plot(epochs, v_acc, linestyle='-.', linewidth=2, color='lime', alpha=0.8, label='Valid Acc')
    plt.xlabel('Epochs', fontsize=11)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(fontsize=12)
    plt.tight_layout()
    image_export_name = "../export/training_data" + export_suffix + ".png"
    plt.savefig(image_export_name, dpi=250)
    plt.show()
