# example of progressively loading images from file
from keras.preprocessing.image import ImageDataGenerator

# create generator
datagen = ImageDataGenerator()

# prepare an iterators for each dataset
train_it = datagen.flow_from_directory('../images/draft-dataset/', class_mode='binary')

# confirm the iterator works
batchX, batchy = train_it.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
