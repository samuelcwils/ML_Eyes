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


# Set path to dataset
data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'

# Get max and min sizes of the images and masks
max_height, max_width, min_height, min_width = shape_maxmin(data_path)

#print(f"Min Height:\t {min_height}") 
#print(f"Min Width:\t {min_width}") 


# Loads all images and masks, resizing to min height and min width
# Return the images and corresponding labels as numpy array tensors
# of size (num_images, min_height, min_width, 3)
def get_ds(data_path, min_height, min_width):

    img_paths = list()
    mask_paths = list()

    # Find all the image files from the path data_path
    for img_path in glob.glob(data_path+"/data/*"):
        img_paths.append(img_path)

    #print("get_ds: Min height:", min_height)
    #print("get_ds: Min width:", min_width)

    #print('Image Paths:')
    #print('img_paths: ', img_paths)

    images = np.zeros((len(img_paths),min_height,min_width,3))

    # The masks were coming out in a different order than the images.
    # Looping over the image paths and loading the corresponding masks.
    #for mask_path in glob.glob(data_path+"/labels/masks/0/*merged.png"):
    for jj in range(len(img_paths)):
        stem = Path(img_paths[jj]).stem
        mask_path = glob.glob(data_path+"/labels/masks/0/"+stem+"_merged.png")
        mask_paths.append(mask_path[0])

    #print('Mask Paths:')
    #print('mask_paths: ', mask_paths)

    masks = np.zeros((len(mask_paths),min_height,min_width,3))

    # Load and resize the images and masks
    for i, img_path in enumerate(img_paths):
        myim = Image.open(img_path)
        images[i] = getPic(img_path, min_height, min_width)

    for i, mask_path in enumerate(mask_paths):
        masks[i] = getPic(mask_path, min_height, min_width)

    return images,masks


# Load and resize an individual image or mask
def getPic(img_path, min_height, min_width):
    # The resize command takes width then height!
    size = min_width, min_height
    myim = Image.open(img_path)

    myim = myim.resize(size)

    #Image._show(myim)
    myimage = asarray(myim)
    return myimage



print(f"Loading dataset and resizing to\t {min_height, min_width}") 
dataX, dataY = get_ds(data_path, min_height, min_width)

print('dataX (images) shape: ', dataX.shape)
print('dataY (masks) shape: ', dataY.shape)


# Test looking at one image from the data set to make sure it is still OK

data = dataX[5]
#print('data shape: ', data.shape)
img = Image.fromarray(np.uint8(data)).convert('RGB')
#print(img.size)
Image._show(img)

data = dataY[5]
#print('data shape: ', data.shape)
img = Image.fromarray(np.uint8(data)).convert('RGB')
#print(img.size)
Image._show(img)

# For some reason the masks and the images seem to be in different
# orders!
