import tensorflow as tf
import keras
from keras import layers

# Making a fairly simple CNN for segmentation.

# 8/24/24: Currently runs and builds the CNN, but losses are huge
# and grow with each epoch. Next: try a different (custom) loss function.


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

# As my U-Net code is currently written, height and width need to be
# powers of 2.
# Consider updating to remove this restriction or at least get these powers of two in an
# automated way instead of hard coded. 
# Currently hard coded to a size that works with the kvasir dataset.

# I shortened the size of the NN by doing 1 conv2d per layer and
# removing some of the layers. Maybe increase again when we have more
# data and when loss performance is better.
def build_unet_model(min_height, min_width, n_filters=64):
    # inputs
    #inputs = layers.Input(shape=(128,128,3))
    # 256x256x3 for kvasir data
    # 64x64x3 for kvasir faster testing
    inputs = layers.Input(shape=(min_height,min_width,3))
    #print('inputs shape = ', inputs.shape)
    # 1 - downsample
    f1, p1 = downsample_block(inputs, n_filters * (2**0))
    # new image size 128
    #print('f1 shape ', f1.shape)
    #print('p1 shape ', p1.shape)
    # 2 - downsample
    f2, p2 = downsample_block(p1, n_filters * (2**1))
    # new image size 64
    #print('f2 shape ', f2.shape)
    #print('p2 shape ', p2.shape)
    # 3 - downsample
    f3, p3 = downsample_block(p2, n_filters * (2**2))
    # new image size 32
    #print('f3 shape ', f3.shape)
    #print('p3 shape ', p3.shape)
    # 4 - downsample
    f4, p4 = downsample_block(p3, n_filters * (2**3))
    # new image size 16
    #print('f4 shape ', f4.shape)
    #print('p4 shape ', p4.shape)
    # 5 - bottleneck
    # bottleneck = single_conv_block(p2, 1024)
    #bottleneck = single_conv_block(p4, 1024)
    bottleneck = double_conv_block(p4, n_filters * (2**4))
    #print('bottleneck shape ', bottleneck.shape)
    # decoder: expanding path - upsample
    # 6 - upsample
    # u6 = upsample_block(bottleneck, f2, 512)
    u6 = upsample_block(bottleneck, f4, n_filters * (2**3))
    # new image size 32
    #print('u6 shape ', u6.shape)
    # 7 - upsample
    # u7 = upsample_block(u6, f1, 256)
    u7 = upsample_block(u6, f3, n_filters * (2**2))
    # new image size 64
    #print('u7 shape ', u7.shape)
    # 8 - upsample
    # u8 = upsample_block(u7, f1, 128)
    u8 = upsample_block(u7, f2, n_filters * (2**1))
    # new image size 128
    #print('u8 shape ', u8.shape)
    # 9 - upsample
    u9 = upsample_block(u8, f1, n_filters * (2**0))
    # new image size 256
    #print('u9 shape ', u9.shape)
        # outputs
    # outputs = (layers.Conv2D(1, 1, padding="same", activation = "sigmoid")(u7))
    outputs = layers.Conv2D(1, 1, padding="same", activation = "sigmoid", dtype="float32")(u9)
    #outputs = layers.Conv2D(1, 1, activation = "softmax")(u9)
    #print('output shape = ', outputs.shape)
    # unet model with Keras Functional API
    unet_model = keras.Model(inputs, outputs, name="U-Net")
    return unet_model