# Load images and determine the max and min heights and widths.
#
# Then later, we can decide whether to expand all images to the same
# size or crop all images to the same size. Go to largest or go to
# smallest.

import PIL
from PIL import Image
import glob



def shape_maxmin(datapath):
    # Load each image and get height and width.
    # Keep track of max and min height and width as we go.
    # Do the same with each mask.

    max_height = 0
    max_width = 0
    min_height = 1e+10
    min_width = 1e+10

    img_paths = list()

    # Loop over all data images
    for img_path in glob.glob(data_path+"/data/*"):
        #img_paths.append(img_path)
        #images = np.zeros((len(img_paths),256,256,3))

        myimage = Image.open(img_path)

        # height
        myheight = myimage.size[0]
        #print(myheight)

        # width
        mywidth = myimage.size[1]
        #print(mywidth)

        if myheight > max_height:
            max_height = myheight

        if mywidth > max_width:
            max_width = mywidth

        if myheight < min_height:
            min_height = myheight

        if mywidth < min_width:
            min_width = mywidth

    # Loop over all masks
    for img_path in glob.glob(data_path+"/labels/masks/0/*merged.png"):

        myimage = Image.open(img_path)

        # height
        myheight = myimage.size[0]
        #print(myheight)

        # width
        mywidth = myimage.size[1]
        #print(mywidth)

        if myheight > max_height:
            max_height = myheight

        if mywidth > max_width:
            max_width = mywidth

        if myheight < min_height:
            min_height = myheight

        if mywidth < min_width:
            min_width = mywidth


    return max_height, max_width, min_height, min_width




data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'

max_height, max_width, min_height, min_width = shape_maxmin(data_path)

#print(f"Max Height:\t {max_height}") 
#print(f"Max Width:\t {max_width}") 
#print(f"Min Height:\t {min_height}") 
#print(f"Min Width:\t {min_width}") 

