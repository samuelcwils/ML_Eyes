# Load all images and masks, resizing everything to the min height and
# width of the dataset files.
# While going through data, we will also augment dataset with random
# rotations, flips, crops etc. using package albumentations.

import PIL
from PIL import Image
import numpy as np
import numpy.ma as ma
from numpy import asarray
from ImageSizes import shape_maxmin
import glob
from matplotlib import pyplot as plt
from matplotlib import cm
import cv2
from pathlib import Path
import torch

import tensorflow as tf
#import matplotlib.pyplot as plt
from functools import partial
import albumentations as A


###
# For now, decide ahead of time how many augmented images/masks to make
# per original. That way I can set the size of my nparray correctly.
numAugmentations = 1

###

# This code was used when testing the get_ds function.

# Set path to dataset
# data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
datapath = '/home/samwilson/ML_Eyes-main/DataImages/kvasir/Kvasir-SEG/images'
maskpath = '/home/samwilson/ML_Eyes-main/DataImages/kvasir/Kvasir-SEG/masks'

# Get max and min sizes of the images and masks
# max_height, max_width, min_height, min_width = shape_maxmin(data_path)
#max_height, max_width, min_height, min_width = shape_maxmin(datapath, maskpath)
min_height = 256
min_width = 256

#print(f"Min Height:\t {min_height}") 
#print(f"Min Width:\t {min_width}") 

###

#def visualize(original_image, augmented_image, original_mask, augmented_mask):
def visualize_old(original_image, original_mask):
    fig = plt.figure()
    plt.subplot(1,4,1)
    plt.title('Original image')
    plt.imshow(original_image)
    #plt.show()

    plt.subplot(1,4,2)
    plt.title('Augmented image')
    flipped = tf.image.flip_left_right(original_image)
    plt.imshow(flipped)
    #plt.show()
    #visualize(image, flipped)

    plt.subplot(1,4,3)
    plt.title('Original mask')
    plt.imshow(original_mask)
    #plt.show()

    plt.subplot(1,4,4)
    plt.title('Augmented mask')
    #plt.imshow(augmented_mask)
    flipped = tf.image.flip_left_right(original_mask)
    plt.imshow(flipped)
    plt.show()
    #visualize(image, flipped)


def visualize(image, mask, original_image=None, original_mask=None):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(2, 1, figsize=(8, 8))

        ax[0].imshow(image)
        ax[1].imshow(mask)
        plt.show()
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('Original mask', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('Transformed image', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('Transformed mask', fontsize=fontsize)
        plt.show()

# Loads all images and masks, resizing to min height and min width.
# Return the images and corresponding labels as np array tensors
# of size (num_images, min_height, min_width, 3).
def get_ds(datapath, maskpath, min_height, min_width):

    img_paths = list()
    mask_paths = list()

    # Find all the image files from the path data_path
    #for img_path in glob.glob(data_path+"/data/*"):
    for img_path in glob.glob(datapath+"/*"):
        img_paths.append(img_path)

    images = np.zeros((len(img_paths)*numAugmentations,min_height,min_width,3))
    #print(images[0].shape)

    # Loop over the image paths and load the corresponding masks.
    #for mask_path in glob.glob(data_path+"/labels/masks/0/*merged.png"):
    for jj in range(len(img_paths)):
        stem = Path(img_paths[jj]).stem
        mask_path = glob.glob(maskpath+"/*")
        #print(mask_path)
        mask_paths.append(mask_path[0])

    masks = np.zeros((len(mask_paths)*numAugmentations,min_height,min_width,3))

    # Load and resize the images and masks.
    for i, img_path in enumerate(img_paths):
        #myim = Image.open(img_path)
        #myim.show()
        newpic = getPic(img_path, min_height, min_width)
        #print('newpic shape ', newpic.shape)
        if newpic.shape == (min_height,min_width): # i.e., if newpic is grayscale
            stacked_pic = np.stack((newpic,)*3, axis=-1)
            images[i] = stacked_pic/255.0
        else:
            #images[i] = getPic(img_path, min_height, min_width)
            images[i] = newpic/255.0
            #images[i] = newpic
        #print('images[i] shape', images[i].shape)
        #img1 = Image.fromarray(images[i,:,:,:].astype('uint8'), 'RGB')
        #img1 = Image.fromarray(images[i].astype('uint8'), mode='RGB')
        #img1 = Image.fromarray(images[i], mode='RGB')
        arr = np.uint8(255 * images[i])
        img2 = Image.fromarray(arr, mode="RGB")
        #img2.show()

    for i, mask_path in enumerate(mask_paths):
        testimage = getPic(mask_path, min_height, min_width)
        masks[i] = testimage
        arr = np.uint8(masks[i])
        img2 = Image.fromarray(arr, mode="RGB")
        # Mask needs to be one-hot encoded for categorical cross-entropy
        # loss function.
        #img2.show()

    return images,masks



# Load and resize an individual image or mask
def getPic(img_path, min_height, min_width):
    # The resize command takes width then height!
    size = min_width, min_height
    myim = Image.open(img_path)
    #myim.show()

    myim = myim.resize(size)

    myimage = asarray(myim)
    #myimage.show()
    # they look OK here (9/2/2024)
    #img = Image.fromarray(myimage, 'RGB')
    #img.save('my.png')
    #img.show()

    return myimage



###
# Check that get_ds works.

dataX, dataY = get_ds(datapath, maskpath, min_height, min_width)
image = dataX[0]
mask = dataY[0]
#visualize(dataX[0], dataY[0])
#aug = A.PadIfNeeded(min_height=128, min_width=128, p=1)
aug = A.ShiftScaleRotate(shift_limit = (-0.25, 0.25), scale_limit = (-0.5, 0.5), p = 1.0) 

augmented = aug(image=image, mask=mask)

image_augmented = augmented['image']
mask_augmented = augmented['mask']

print(image_augmented.shape, mask_augmented.shape)

visualize(image_augmented, mask_augmented, original_image=image, original_mask=mask)


#visualize(dataX[0], dataY[0], dataX[0], dataY[0])



# test visualization




#print('dataX (images) shape: ', dataX.shape)
#print('dataY (masks) shape: ', dataY.shape)

###


# Looking at one image from the data set to make sure it is still OK

#data = dataX[1]
#print('data shape: ', data.shape)
#img = Image.fromarray(np.uint8(data)).convert('RGB')
#print(img.size)
#Image._show(img)

#data = dataY[1]
#print('data shape: ', data.shape)
#img = Image.fromarray(np.uint8(data)).convert('RGB')
#print(img.size)
#Image._show(img)

