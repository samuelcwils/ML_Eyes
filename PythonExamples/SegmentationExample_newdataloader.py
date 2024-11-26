# Making a fairly simple CNN for segmentation.

# 8/24/24: Currently runs and builds the CNN, but losses are huge
# and grow with each epoch. Next: try a different (custom) loss function.

import keras
from keras import layers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from NeptunePredictionsLogger import NeptunePredictionsLogger
from TensorflowLoadDataset import get_tensorflow_dataset

# Set seed for randomization so we can repeat runs
# seed = 300507
# keras.utils.set_random_seed(seed)
# print('seed ', seed)

#tf.config.experimental.enable_op_determinism()
#Determinism breaks things for me. Has to do with the maxpool operation. I think it's CUDA. - Sam

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

train_batch_size = 8
valid_batch_size = 8
test_batch_size = 8

dataset = get_tensorflow_dataset((min_width, min_height), data_path, mask_path) 
train_dataset = dataset.take(len(dataset) * 8 // 10)
valid_dataset = dataset.skip(len(dataset) * 8 // 10)

test_dataset = valid_dataset.skip(len(valid_dataset) // 2)
valid_dataset = valid_dataset.take(len(valid_dataset) // 2)

#all this take and skip stuff allocates the images to the different datasets train, valid, and test

train_dataset = train_dataset.batch(train_batch_size).prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(valid_batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(test_batch_size).prefetch(tf.data.AUTOTUNE)


# Using U-net as an example CNN architecture.
# U-net is useful for segmentation.
# https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/


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
                  metrics=['categorical_accuracy'])
                  #metrics=['accuracy', 'categorical_accuracy'])
                  #metrics=['accuracy', 'mse'])

# Output the model summary. This shows the sizes of input and output at
# each layer and number of parameters to be trained etc. 
unet_model.summary()

#Change this to use your own account's api token
# run = neptune.init_run(project='knightenjoyer15/Project', api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTE1ZGJhZi1jOTMyLTRiM2QtYTY3MC1jYzJlZTYyYTczODEifQ==")

# predictions_neptune_callback = NeptunePredictionsLogger(run, unet_model, test_dataset)
# normal_neptune_callback = NeptuneCallback(run=run, log_on_batch=True)

# Train the model.
unet_model.fit(
    x=train_dataset,
    epochs=3,
    verbose="auto",
    #callbacks=[predictions_neptune_callback, normal_neptune_callback],
    validation_data=valid_dataset,
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
score = unet_model.evaluate(test_dataset, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
