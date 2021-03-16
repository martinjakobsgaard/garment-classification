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


# Fetch data
def data_import(argument_path, argument_size, argument_resolution, argument_class_count):
    print("\n=== IMPORT DATA =======================\n")
    print("Loading data...", end=' ')
    train_data_generator = ImageDataGenerator()
    train_data_iterator = train_data_generator.flow_from_directory(
        argument_path,
        color_mode='rgb',
        target_size=(argument_resolution, argument_resolution),
        class_mode='sparse',
        batch_size=argument_size)
    print("Train data batch size: ", train_data_iterator.batch_size)
    print("Number of classes: ", train_data_iterator.num_classes)
    train_images, train_labels = train_data_iterator.next()

    train_images = tf.keras.applications.resnet50.preprocess_input(train_images)
    train_labels_categorical = tf.keras.utils.to_categorical(train_labels, num_classes=argument_class_count, dtype='uint8')
    print("Done!")

    # Split datasets
    print("Splitting datasets...")
    train_images, validation_images, train_labels, validation_labels = train_test_split(
        train_images,
        train_labels_categorical,
        test_size=0.20,
        stratify=train_labels_categorical,
        random_state=40,
        shuffle=True)
    print("Done!")

    # Generate batches
    print("Generating batches...")
    train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.1,
        width_shift_range=0.01,
        height_shift_range=0.01,
        horizontal_flip=False)

    validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
    result_train_image_set = train_data_generator.flow(train_images, train_labels, batch_size=setting_batch_size)
    result_validation_image_set = validation_data_generator.flow(validation_images, validation_labels, batch_size=setting_batch_size)
    print("Done!")

    return result_train_image_set, result_validation_image_set


# Defining backbone
def define_backbone_resnet50(argument_resolution, argument_class_count, argument_frozen_min, argument_frozen_max):
    print("\n=== DEFINE BACKBONE ======================\n")
    print("Defining input shape...")
    input_shape = K.Input(shape=(argument_resolution, argument_resolution, 3))

    print("Define ResNet50...")
    # Source: https://keras.io/api/applications/resnet/#resnet50-function
    result_backbone = K.applications.ResNet50(include_top=False, weights="imagenet", input_tensor=input_shape, pooling=max, classes=argument_class_count) # <- Good input

    # If freeze layers in model
    if argument_frozen_min != argument_frozen_max:
        if argument_frozen_max == -1:
            print("Freezing layer " + str(argument_frozen_min) + " -> max!")
            for layer in result_backbone.layers[argument_frozen_min:]:
                layer.trainable = False
        else:
            print("Freezing layer " + str(argument_frozen_min) + " -> " + str(argument_frozen_max) + '!')
            for layer in result_backbone.layers[argument_frozen_min:argument_frozen_max]:
                layer.trainable = False
    else:
        print("Not freezing any layers...")

    return result_backbone


# Build model
def build_model(argument_backbone, argument_resolution, argument_class_count):
    print("Defining model...")
    to_res = (argument_resolution, argument_resolution)
    result_model = K.models.Sequential()
    result_model.add(K.layers.Lambda(lambda image: tf.image.resize(image, to_res)))
    #model.add(res_model)
    result_model.add(argument_backbone)
    result_model.add(K.layers.Flatten())
    result_model.add(K.layers.BatchNormalization())
    result_model.add(K.layers.Dense(256, activation='relu'))
    result_model.add(K.layers.Dropout(0.5))
    result_model.add(K.layers.BatchNormalization())
    result_model.add(K.layers.Dense(128, activation='relu'))
    result_model.add(K.layers.Dropout(0.5))
    result_model.add(K.layers.BatchNormalization())
    result_model.add(K.layers.Dense(64, activation='relu'))
    result_model.add(K.layers.Dropout(0.5))
    result_model.add(K.layers.BatchNormalization())
    result_model.add(K.layers.Dense(argument_class_count, activation='softmax'))

    result_checkpoint = K.callbacks.ModelCheckpoint(filepath="../models/" + global_export_suffix + "-checkpoint.h5", monitor="val_accuracy", mode="max", save_best_only=True)

    # Compile model
    print("Compiling model...")
    result_model.compile(loss='categorical_crossentropy', optimizer=K.optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

    return result_model, result_checkpoint


# Plot data
def plot_history(model_history):
    print("\n=== PLOT MODEL ========================\n")
    loss = model_history.history['loss']
    v_loss = model_history.history['val_loss']
    acc = model_history.history['accuracy']
    v_acc = model_history.history['val_accuracy']
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
    image_export_name = "../export/" + global_export_suffix + "_training_graph.png"
    plt.savefig(image_export_name, dpi=250)
    plt.show()


if __name__ == '__main__':
    # Settings
    setting_title = "RESNET50-NO-LINEN"
    setting_resolution = 224
    setting_batch_size = 64
    setting_epochs = 50
    setting_frozen_min = 0
    setting_frozen_max = -1  # max
    setting_class_count = 4
    setting_dataset_size = 7029
    setting_path = '../images/garment-dataset-2-no-linen/train/'

    # Generate export name unique to training
    global_export_suffix = setting_title
    global_export_suffix += ("_res" + str(setting_resolution))
    global_export_suffix += ("_batch" + str(setting_batch_size))
    global_export_suffix += ("_epoch" + str(setting_epochs))
    global_export_suffix += ("_frozen" + str(setting_frozen_min) + "-" + str(setting_frozen_max))
    global_export_suffix += ("_class" + str(setting_class_count))
    global_export_suffix += ("_size" + str(setting_dataset_size))

    # Create model
    train_image_set, validation_image_set = data_import(setting_path, setting_dataset_size, setting_resolution, setting_class_count)
    backbone = define_backbone_resnet50(setting_resolution, setting_class_count, setting_frozen_min, setting_frozen_max)
    model, checkpoint = build_model(backbone, setting_resolution, setting_class_count)

    # Final prompt
    response = input("Do you want to start training? (Y/n) ")
    if response == "n":
        exit(0)

    # Train model
    model_train = model.fit(
        train_image_set,
        batch_size=setting_batch_size,
        epochs=setting_epochs,
        verbose=1,
        validation_data=validation_image_set,
        callbacks=[checkpoint])
    model.summary()

    # Optional: plot model
    plot_history(model_train)

    # Export model
    model_export_name = "../models/" + global_export_suffix + ".h5"
    model.save(model_export_name, save_format='h5')

