# This code reads in all mask files for each given image file in a data
# directory and merges the masks into one mask. 
# Once run, we should have exactly the same number of images and masks.

import PIL
from PIL import Image
import numpy as np
import numpy.ma as ma
from numpy import asarray
import cv2
import glob
from pathlib import Path



# Test of merging two masks:

## Load two masks and convert mask file to binary 0/1 if necessary 
#img1 = cv2.imread('/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train/labels/masks/0/01775d1c82e56c4d_m09ddx_eededdef.png', 2)
#img2 = cv2.imread('/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train/labels/masks/0/01775d1c82e56c4d_m09ddx_bac9d296.png', 2)
  
#ret, bw_img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY) 
#ret, bw_img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY) 
  
## Convert to binary form 
#bw1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY) 
#bw2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY) 

#waittime = 2000

#cv2.imshow("Binary", bw_img1) 
## close window after waittime ms
#cv2.waitKey(waittime) 

#cv2.imshow("Binary", bw_img2) 
#cv2.waitKey(waittime) 

#print('bw_img2 shape')
#print(bw_img2.shape)
#print(f"Height:\t\t {bw_img2.shape[0]}") 
#print(f"Width:\t\t {bw_img2.shape[1]}")

## Convert to np arrays
#mydata1 = asarray(bw_img1)
#mydata2 = asarray(bw_img2)

## Make new image mask with logical OR
#newdata = mydata1 | mydata2

## Convert back to image
#mynewimage = Image.fromarray(newdata)

#mynewimage.show()


def merge_masks(data_path):
    # Load an image
    # Load all masks associated with that image
    # Merge the masks into one mask
    # Save the merged mask to a new file: same file stem with _merged
    # appended.

    img_paths = list()

    for img_path in glob.glob(data_path+"/data/*"):
        img_paths.append(img_path)
    images = np.zeros((len(img_paths),256,256,3))

    # This gets the filename stem from the full path.
    # Loop over img_paths,
    # Get list of masks that go with that image
    # Merge the masks into one mask
    # Save the new mask

    img_stems = list()
    for i, img_path in enumerate(img_paths):
        #print(Path(img_paths[i]).stem)
        img_stems.append(Path(img_paths[i]).stem)
        #images[i] = getPic(img_path)


    for j, stem in enumerate(img_stems):
        mask_paths = list()
        #print(Path(img_paths[j]).stem)

        # get all masks for current image stem
        for mask_path in glob.glob(data_path+"/labels/masks/0/"+stem+"*"):
            mask_paths.append(mask_path)

        # Convert to np arrrays and combine
        # Load masks and convert mask file to binary 0/1 if necessary 

        if len(mask_paths) == 0:
            print(stem)
        
        if len(mask_paths) > 0:
            img1 = cv2.imread(mask_paths[0])
            ret, bw_img1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY) 

            # Convert to binary form 
            bw1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY) 

            newmask = asarray(bw_img1)


            for k, currentpath in enumerate(mask_paths):
                img2 = cv2.imread(mask_paths[k])
                ret, bw_img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY) 
                bw2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY) 
    
                nextmask = asarray(bw_img2)

                newmask = newmask | nextmask

            newMaskImage = Image.fromarray(newmask)
            #newMaskImage.show()

            gr_im = newMaskImage.save(data_path+"/labels/masks/0/"+stem+"_merged.png")



#data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train'

merge_masks(data_path)


