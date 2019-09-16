from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

# build CNN

# Initialising CNN
classifier = Sequential()
# step 1 - Convolution
classifier.add(Convolution2D(input_shape=(128,128,3), data_format="channels_last" , filters=32, kernel_size=(3,3), activation="relu", kernel_initializer="uniform"))
# step 2 - Max pooling
classifier.add(MaxPool2D(pool_size=(2,2), strides = 2))

classifier.add(Convolution2D(data_format="channels_first", filters=16, kernel_size = (3, 3), activation="relu", kernel_initializer="uniform"))
classifier.add(MaxPool2D(pool_size=(2,2), strides = 2))

# step 3 - flatten
classifier.add(Flatten())
# step 4 - using ANN to categorize
classifier.add(Dense(64, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(16, activation="relu", kernel_initializer="uniform"))
classifier.add(Dense(1, activation="sigmoid", kernel_initializer="uniform"))

# compile and fit
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# image preproccessing
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        train_generator,
        steps_per_epoch = 8000,
        epochs = 25,
        validation_data = validation_generator,
        validation_steps = 2000)