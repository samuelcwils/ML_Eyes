# Load all images and masks, resizing everything to the min height and
# width of the dataset files.

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


###

# This code was used when testing the get_ds function.

# Set path to dataset
# data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
#datapath = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/images_small'
#maskpath = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/masks_small'

# Get max and min sizes of the images and masks
# max_height, max_width, min_height, min_width = shape_maxmin(data_path)
#max_height, max_width, min_height, min_width = shape_maxmin(datapath, maskpath)
min_height = 256
min_width = 256

#print(f"Min Height:\t {min_height}") 
#print(f"Min Width:\t {min_width}") 

###

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

    images = np.zeros((len(img_paths),min_height,min_width,3))
    #print(images[0].shape)

    # Loop over the image paths and load the corresponding masks.
    #for mask_path in glob.glob(data_path+"/labels/masks/0/*merged.png"):
    for jj in range(len(img_paths)):
        stem = Path(img_paths[jj]).stem
        mask_path = glob.glob(maskpath+"/*")
        #print(mask_path)
        mask_paths.append(mask_path[0])

    masks = np.zeros((len(mask_paths),min_height,min_width,3))

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

#dataX, dataY = get_ds(datapath, maskpath, min_height, min_width)

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

