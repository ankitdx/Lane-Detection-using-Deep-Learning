import sys
sys.path.append('/content/drive/My Drive/Colab Notebooks/LaneDetection')
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
from LaneDetection.For_github.VGG import vgg16_unet
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define Directories to read numpy arrays
data_dir = '/content/drive/My Drive/Colab Notebooks/Datasets/LaneDetection/'

# Define Model Path
pretrained = '/content/drive/My Drive/Colab Notebooks/Models/LaneDetector_vgg.h5'
path2save = '/content/drive/My Drive/Colab Notebooks/Models/LaneDetector_MeanIOU.h5'

width = 640
height = 384
depth = 3

# Load numpy arrays from the directory
X_train = np.load(data_dir+'train_data.npz')['arr_0']
Y_train = np.load(data_dir+'train_data_labels.npz')['arr_0']
X_test = np.load(data_dir+'test_data.npz')['arr_0']
Y_test = np.load(data_dir+'test_data_labels.npz')['arr_0']

# Create Network Graph in the Memory
net = vgg16_unet(height, width, depth, pretrained)


# Store the network weights whenever loss is minimum than previous epoch
modelCheck = callbacks.ModelCheckpoint(path2save, monitor='loss',
                                       verbose=0, save_best_only=True, mode='min')

# Let's create Data generators for image and mask augmentation
# for Train and Test set
train_image_datagen = ImageDataGenerator(rescale=1./255,
                                         shear_range=0.3,
                                         zoom_range=0.3,
                                         rotation_range=90,
                                         width_shift_range=0.3,
                                         horizontal_flip=True,
                                         height_shift_range=0.3)

train_mask_datagen =  ImageDataGenerator(shear_range=0.3,
                                         zoom_range=0.3,
                                         rotation_range=90,
                                         width_shift_range=0.3,
                                         horizontal_flip=True,
                                         height_shift_range=0.3)

seed = 1

train_image_gen = train_image_datagen.flow(X_train, batch_size=8,
                                           seed=seed, shuffle=False)
train_label_gen = train_mask_datagen.flow(Y_train, batch_size=8,
                                          seed=seed, shuffle=False)

# No augmentation on Test data set
test_image_datagen = ImageDataGenerator(rescale=1/.255)
test_mask_datagen =  ImageDataGenerator()


test_image_gen = test_image_datagen.flow(X_test, batch_size=8,
                                         seed=seed, shuffle=True)
test_label_gen = test_mask_datagen.flow(Y_test, batch_size=8,
                                        seed=seed, shuffle=True)

# Zip images and mask generators
train_generator = zip(train_image_gen, train_label_gen)
test_generator = zip(test_image_gen, test_label_gen)


# Define our dice coeff
def dice_coef(y_true, y_pred, smooth=1):
    import tensorflow.keras.backend as K
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


print ("Compiling Model...")
# Set the compiler parameter for the training
opt = Adam(lr=1e-7)
net.compile(loss="binary_crossentropy", optimizer=opt,
            metrics=[dice_coef])

print ("Training the Model...")
# Train the Network
batch_size = 6
epochs = 500
net.fit_generator(train_generator, steps_per_epoch=len(X_train)//batch_size,
                  epochs=500, callbacks=[modelCheck],
                  validation_data=test_generator,
                  validation_steps=len(X_test)//batch_size)