import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.losses import categorical_crossentropy

from keras.preprocessing.image import ImageDataGenerator

from scipy import ndimage
from sklearn.externals._pilutil import imresize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import matplotlib.pyplot as plt

import cv2


# image augmentation techniques

def display_side_by_side(img1, img2):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img1, cmap='gray')
    ax[0].axis('off')
    ax[1].imshow(img2, cmap='gray')
    ax[1].axis('off')


def ogrid(img):
    nimg = np.copy(img)
    lx, ly = nimg.shape
    X, Y = np.ogrid[0:lx, 0:ly]
    mask = (X - lx / 2) ** 2 + (Y - ly / 2) ** 2 > lx * ly / 4
    nimg[mask] = 0
    return nimg


def rotate_img(img):
    angle = np.random.choice(np.random.uniform(-45, 45, 100))
    nimg = np.copy(img)
    nimg = ndimage.rotate(nimg, angle)
    height, width = img.shape
    nimg = imresize(nimg, (width, height))
    return nimg


def blur_img(img):
    nimg = np.copy(img)
    sigma = np.random.randint(1, 2)
    blurred_img = ndimage.gaussian_filter(nimg, sigma=sigma)
    return blurred_img


def flip_img(img):
    nimg = np.copy(img)
    nimg = np.fliplr(nimg)
    return nimg


def add_noise(img):
    nimg = np.copy(img)
    noise = np.random.normal(0, 0.5, size=(48, 48)).astype(np.uint8) * 255
    nimg += noise
    nimg = np.clip(nimg, 0, 255)
    return nimg


def augment_img(img):
    methods = [ogrid, rotate_img, blur_img, flip_img, add_noise]

    method = np.random.choice(methods)
    return method(img)


# pixels is the series from the Dataframe
def extract_from_string(pixels):
    pixels = pixels.split(' ')
    pixels = np.array([int(i) for i in pixels])
    return np.reshape(pixels, (48, 48))


def extract_image(pixels):
    pixels = pixels.as_matrix()[0]  # The output is a string
    return extract_from_string(pixels)


# --------------------START------------------------

df = pd.read_csv('./fer2013.csv')
# print(df.info())

value_counts = df['emotion'].value_counts().sort_index()
max_value = df['emotion'].value_counts().max()
max_idx = df['emotion'].value_counts().idxmax()

new_df = pd.DataFrame()
for i, row in df.iterrows():
    # Take this row and convert its pixel type to actual image
    new_df = new_df.append(
        pd.Series([row.emotion, extract_from_string(row.pixels), row.Usage], index=['emotion', 'pixels', 'Usage'],
                  name=str(i)))

print("new_df")
print(new_df)

augmented_df = new_df.copy()
unique_emotions = new_df.emotion.unique()
for emotion in unique_emotions:
    if emotion != max_idx:
        # This is the dataset we want to augment
        # Find the current length of this emotion
        emotion_df = augmented_df[augmented_df.emotion == emotion]
        current_size = len(emotion_df)
        images_2_generate = max_value - current_size
        for i in range(0, images_2_generate):
            # Choose a random image
            emotion_df = augmented_df[augmented_df.emotion == emotion].sample(n=1)
            current_img = emotion_df.pixels[0]
            nimg = augment_img(current_img)

            # Add a new row
            row = pd.Series([emotion, nimg, "Training"], index=["emotion", "pixels", "Usage"], name=str(i))
            augmented_df = augmented_df.append(row)
#new start
train_df = augmented_df[augmented_df.Usage == "Training"]
test_df = augmented_df[augmented_df.Usage == "PrivateTest"]

trainData = np.array(train_df.pixels, dtype=pd.Series)
trainLabels = np.array(train_df.emotion, dtype=pd.Series)
trainLabels = np_utils.to_categorical(trainLabels, 7)

testData = np.array(train_df.pixels, dtype=pd.Series)
testLabels = np.array(train_df.emotion, dtype=pd.Series)
testLabels = np_utils.to_categorical(testLabels, 7)

td = []
for t in trainData:
    t = np.reshape(t, (48, 48, 1))
    td.append(t)

tl = []
for t in trainLabels:
    tl.append(t)

trainData = np.array(td)
trainLabels = np.array(tl)

trainData, trainLabels = shuffle(trainData, trainLabels)

td = []
for t in testData:
    t = np.reshape(t, (48, 48, 1))
    td.append(t)

tl = []
for t in testLabels:
    tl.append(t)

testData = np.array(td)
testLabels = np.array(tl)
#new end
num_labels = 7
batch_size = 32
epochs = 30

model = Sequential()
# First set Conv Layers
model.add(Conv2D(64, (3, 3), padding='valid', input_shape=(48, 48, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

# 2nd set Conv layers
model.add(Conv2D(128, (3, 3), padding='valid', input_shape=(48, 48, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (3, 3), padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# Set of FC => Relu layers
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Softmax classifier
model.add(Dense(7))
model.add(Activation('softmax'))

print("MODEL SUMMARY: ")
print(model.summary())

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Data Augmentation
# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
#     zoom_range=0.1,  # Randomly zoom image
#     width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#     # height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False)  # randomly flip images
#
# datagen.fit(X_train)
trainData, validationData, trainLabels, validationLabels = train_test_split(trainData, trainLabels, test_size=0.2, random_state=20)
# Training the model
model.fit(trainData, trainLabels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(validationData, validationLabels),
          shuffle=True)

# Saving the  model to  use it later on
fer_json = model.to_json()
with open("AIModel3.json", "w") as json_file:
    json_file.write(fer_json)
model.save_weights("AIModel3.h5")
