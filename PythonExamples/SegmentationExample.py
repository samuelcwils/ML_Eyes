# Making a fairly simple CNN for segmentation.

# 8/24/24: Currently runs and builds the CNN, but losses are huge
# and grow with each epoch. Next: try a different (custom) loss function.

from ImageSizes import shape_maxmin
from LoadDataset import get_ds
import keras
from keras import layers
import keras_cv
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


# Set path for dataset.
# Assumes the following structure:
#   data images in data_path/data/
#   corresponding masks in data_path/labels/masks/0/

#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train'
data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/images'
mask_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/masks'


# Merge multiple mask files into a single (if not already done).
# Already merged in this case.


# Load dataset (images and masks all resized to same size and stored in
# an np array tensor).
max_height, max_width, min_height, min_width = shape_maxmin(data_path, mask_path)

#mysize = np.minimum(min_height, min_width)

# Height and width need to be powers of 2 for some reason?
# Need to remove this restriction or get these powers of two in an
# automated way instead of hard coded.
#min_height = 512
min_height = 256
min_width = 256

print('Using image height ', min_height)
print('Using image width ', min_width)

dataX, dataY = get_ds(data_path, mask_path, min_height, min_width)
print('dataX (images) shape: ', dataX.shape)
print('dataY (masks) shape: ', dataY.shape)

# Should really pull these out at random, not in order
# I think there is a shuffle command I can use first?
trainX = dataX[0:799,:,:,:]
trainY = dataY[0:799,:,:,:]
validX = dataX[800:899,:,:,:]
validY = dataY[800:899,:,:,:]
testX = dataX[900:999,:,:,:]
testY = dataY[900:999,:,:,:]

# Using U-net as an example CNN architecture.
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/


# Loop this over images to do augmentation
#def augment(input_image, input_mask):
   #if tf.random.uniform(()) > 0.5:
       ## Random flipping of the image and mask
       #input_image = tf.image.flip_left_right(input_image)
       #input_mask = tf.image.flip_left_right(input_mask)
   #return input_image, input_mask

# Loop this over images to normalize
#def normalize(input_image, input_mask):
   #input_image = tf.cast(input_image, tf.float32) / 255.0
   #input_mask -= 1
   #return input_image, input_mask


def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    print('f shape ', f.shape)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    # print('x shape ', x.shape)
    # print('conv_features shape ', conv_features.shape)
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model(min_height, min_width):
    # inputs
    #inputs = layers.Input(shape=(128,128,3))
    inputs = layers.Input(shape=(min_height,min_width,3))
    #print('inputs shape = ', inputs.shape)
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    #print('f1 shape ', f1.shape)
    #print('p1 shape ', p1.shape)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    #print('f2 shape ', f2.shape)
    #print('p2 shape ', p2.shape)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    #print('f3 shape ', f3.shape)
    #print('p3 shape ', p3.shape)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    #print('f4 shape ', f4.shape)
    #print('p4 shape ', p4.shape)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    #print('bottleneck shape ', bottleneck.shape)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    #print('u6 shape ', u6.shape)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    #print('u7 shape ', u7.shape)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    #print('u8 shape ', u8.shape)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    #print('u9 shape ', u9.shape)
        # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    #outputs = layers.Conv2D(3, 1, activation = "softmax")(u9)
    #print('output shape = ', outputs.shape)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net")
    return unet_model


unet_model = build_unet_model(min_height, min_width)

unet_model.compile(optimizer = keras.optimizers.AdamW(learning_rate=0.0001),
                  loss = "categorical_crossentropy",
                  metrics=['accuracy'])
                  #metrics=['accuracy', 'categorical_accuracy'])
                  #metrics=['accuracy', 'mse'])

unet_model.summary()

unet_model.fit(
    x=trainX,
    y=trainY,
    batch_size=50,
    epochs=5,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=(validX, validY),
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=20,
    validation_freq=1,
)


# Set up a test CNN with Keras.

#num_classes = 2
#input_shape = (min_height, min_width, 3)

# Starting with a pretrained model
#model = keras_cv.models.DeepLabV3Plus.from_preset(
    #"deeplab_v3_plus_resnet50_pascalvoc",
    #num_classes = 2,
    #input_shape = (min_height, min_width, 3),
#)

#model = keras.Sequential(
    #[
        #keras.Input(shape=input_shape),
        #layers.Conv2D(32, strides=1, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        #layers.MaxPooling2D(pool_size=(2, 2)),
        #layers.Flatten(),
        #layers.Dropout(0.5),
        ##layers.Dense(num_classes, activation="softmax"),
        ##layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        ##layers.MaxPooling2D(pool_size=(2, 2)),
        ##layers.UpSampling2D(interpolation='bilinear')
        #layers.UpSampling2D(2)
        #layers.Conv2D(1, kernel_size=1)
    #]
#)

#model.summary()

#batch_size = 128
#epochs = 15

#model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(dataX, dataY, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# Evaluate the trained model

#score = model.evaluate(x_test, y_test, verbose=0)
#print("Test loss:", score[0])
#print("Test accuracy:", score[1])
