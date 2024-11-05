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
import torch


# Set seed for randomization so we can repeat runs
seed = 300007
keras.utils.set_random_seed(seed)
print('seed ', seed)

tf.config.experimental.enable_op_determinism()


# Set path for dataset.
# Assumes the following structure:
#   data images in data_path/data/
#   corresponding masks in data_path/labels/masks/0/

# Old dataset. These were problematic due to mask and image sizes not
# being the same. 
#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train'

# This dataset works better and the images and masks have the same size.
data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/images'
mask_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/masks'

# For quick testing, I made a smaller subset of data.
#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/images_small'
#mask_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/masks_small'

# Merge multiple mask files into a single (if not already done).
# Already merged in this case.
# The kvasir data set did not have the problem of having multiple masks
# per image, and that should also not be a problem with our real pig
# data. 


# Load dataset (images and masks all resized to same size and stored in
# an np array tensor).
max_height, max_width, min_height, min_width = shape_maxmin(data_path, mask_path)

#mysize = np.minimum(min_height, min_width)

# As my U-Net code is currently written, height and width need to be
# powers of 2.
# Consider updating to remove this restriction or at least get these powers of two in an
# automated way instead of hard coded. 
# Currently hard coded to a size that works with the kvasir dataset.

# For kvasir, images can be up to 256x256 to match smallest images.
# Making smaller images for faster testing. 
#min_height = 64
#min_width = 64
min_height = 256
min_width = 256

print('Using image height ', min_height)
print('Using image width ', min_width)

dataX, dataY = get_ds(data_path, mask_path, min_height, min_width)
print('dataX (images) shape: ', dataX.shape)
print('dataY (masks) shape: ', dataY.shape)

# Shuffle the indices in case there is any order to the downloaded data. 
indices = np.arange(dataX.shape[0])
np.random.shuffle(indices)
# Apply the shuffled indices to the data
shuffled_X = dataX[indices]
shuffled_Y = dataY[indices]

# Split the dataset into training, validation, and testing sets.
# This is currently hardcoded to having 1000 images and masks. 
# Update to pull out percentages of dataset for each category.
trainX = shuffled_X[0:799,:,:,:]
trainY = shuffled_Y[0:799,:,:,:]
validX = shuffled_X[800:899,:,:,:]
validY = shuffled_Y[800:899,:,:,:]
testX = shuffled_X[900:999,:,:,:]
testY = shuffled_Y[900:999,:,:,:]


# Using U-net as an example CNN architecture.
# U-net is useful for segmentation.
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/


# Loop this over images to do augmentation.
# This is not done yet.
#def augment(input_image, input_mask):
   #if tf.random.uniform(()) > 0.5:
       ## Random flipping of the image and mask
       #input_image = tf.image.flip_left_right(input_image)
       #input_mask = tf.image.flip_left_right(input_mask)
   #return input_image, input_mask

# Loop this over images to normalize
# This is not done yet. 
#def normalize(input_image, input_mask):
   #input_image = tf.cast(input_image, tf.float32) / 255.0
   #input_mask -= 1
   #return input_image, input_mask


# Do a set of two convolutions, each followed by relu.
# Note that this is using the Keras functional API.
def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

# Similar code using just one convolution for a smaller CNN: faster
# testing and fewer parameters to train.
def single_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
    return x

# Use conv2d -> pooling -> dropout for each layer in the downward part
# of the "U". Num outputs < Num inputs for each later (from the pooling).
def downsample_block(x, n_filters):
    f = single_conv_block(x, n_filters)
    #f = double_conv_block(x, n_filters)
    #print('f shape ', f.shape)
    p = layers.MaxPool2D(2)(f)
    # set dropout
    p = layers.Dropout(0.1)(p)
    return f, p

# Use transposed convolution to come back up the "U". Since we are
# building a mask, need to come back up so final output is same size as
# original image / mask input.
def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    # print('x shape ', x.shape)
    # print('conv_features shape ', conv_features.shape)
    x = layers.concatenate([x, conv_features])
    # set dropout
    x = layers.Dropout(0.1)(x)
    # Conv2D with ReLU activation
    x = single_conv_block(x, n_filters)
    #x = double_conv_block(x, n_filters)
    return x

# customizable loss function for unet network
def bitmask_loss_fn(y_true, y_pred):

    # tweakable parameters
    lambda_L = 5.0
    lambda_S = 0.5

    # compute the loss and return that value
    weight_matrix = tf.where(y_true - y_pred < 0, lambda_L, lambda_S)
    return tf.math.reduce_mean(tf.abs(weight_matrix * (y_true - y_pred)))

# I shortened the size of the NN by doing 1 conv2d per layer and
# removing some of the layers. Maybe increase again when we have more
# data and when loss performance is better.
def build_unet_model(min_height, min_width):
    # inputs
    #inputs = layers.Input(shape=(128,128,3))
    # 256x256x3 for kvasir data
    # 64x64x3 for kvasir faster testing
    inputs = layers.Input(shape=(min_height,min_width,3))
    #print('inputs shape = ', inputs.shape)
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # new image size 128
    #print('f1 shape ', f1.shape)
    #print('p1 shape ', p1.shape)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # new image size 64
    #print('f2 shape ', f2.shape)
    #print('p2 shape ', p2.shape)
    # 3 - downsample
    # f3, p3 = downsample_block(p2, 256)
    # new image size 32
    #print('f3 shape ', f3.shape)
    #print('p3 shape ', p3.shape)
    # 4 - downsample
    # f4, p4 = downsample_block(p3, 512)
    # new image size 16
    #print('f4 shape ', f4.shape)
    #print('p4 shape ', p4.shape)
    # 5 - bottleneck
    bottleneck = single_conv_block(p2, 1024)
    #bottleneck = single_conv_block(p4, 1024)
    #bottleneck = double_conv_block(p4, 1024)
    #print('bottleneck shape ', bottleneck.shape)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f2, 512)
    #u6 = upsample_block(bottleneck, f4, 512)
    # new image size 32
    #print('u6 shape ', u6.shape)
    # 7 - upsample
    u7 = upsample_block(u6, f1, 256)
    #u7 = upsample_block(u6, f3, 256)
    # new image size 64
    #print('u7 shape ', u7.shape)
    # 8 - upsample
    # u8 = upsample_block(u7, f1, 128)
    #u8 = upsample_block(u7, f2, 128)
    # new image size 128
    #print('u8 shape ', u8.shape)
    # 9 - upsample
    # u9 = upsample_block(u8, f1, 64)
    # new image size 256
    #print('u9 shape ', u9.shape)
        # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation = "sigmoid")(u7)
    #outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
    #outputs = layers.Conv2D(3, 1, activation = "softmax")(u9)
    #print('output shape = ', outputs.shape)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net")
    return unet_model


# Build the u-net model.
unet_model = build_unet_model(min_height, min_width)

unet_model.compile(optimizer = keras.optimizers.AdamW(learning_rate=0.0001),
                  loss = "categorical_crossentropy",
                  metrics=['accuracy'])
                  #metrics=['accuracy', 'categorical_accuracy'])
                  #metrics=['accuracy', 'mse'])

# Output the model summary. This shows the sizes of input and output at
# each layer and number of parameters to be trained etc. 
unet_model.summary()

# Train the model.
unet_model.fit(
    x=trainX,
    y=trainY,
    batch_size=50,
    epochs=3,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    #validation_split=0.2,
    #validation_data=None,
    #validation_batch_size=None,
    validation_data=(validX, validY),
    validation_batch_size=20,
    shuffle=False,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_freq=1,
)

# check seed after
#check_seed = torch.random.initial_seed()
#print('check seed after code', check_seed)

    #validation_data=(validX, validY),
    #validation_batch_size=20,

#num_classes = 2
#input_shape = (min_height, min_width, 3)


# Evaluate the trained model

# Model evaluation (with the test data).
score = unet_model.evaluate(testX, testY, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
