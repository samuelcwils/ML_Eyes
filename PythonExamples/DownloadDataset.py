# need to have installed ultralytics (for YOLO), fiftyone (for data visualization)

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from ultralytics import YOLO
import fiftyone as fo
import fiftyone.zoo as foz


# IMPORTANT NOTE (VH 10-21-2024): the datasets from open-images-v7 are problematic. They are the
# ones that have issues with the images and masks not being the same
# size. Instead, I have been using data from kvasir:
# kvasir colon polyp data downloaded from their website:
# https://datasets.simula.no/kvasir-seg/
# (So there is no need to use this code.) 

#dataset = foz.load_zoo_dataset("quickstart")
#dataset = tfds.load(‘open_images/v7’, split='train')
#session = fo.launch_app(dataset)

IMAGES = 1000
CLASSES = ["Giraffe"]
FOLDER = '/Users/vhowle/Projects/ML_Eyes/DataImages/Giraffe'

# get training data
dataset = fo.zoo.load_zoo_dataset("open-images-v7", split="train", classes=CLASSES, max_samples=IMAGES, dataset_dir=FOLDER)

# get testing data
dataset = fo.zoo.load_zoo_dataset("open-images-v7", split="test", classes=CLASSES, max_samples=IMAGES, dataset_dir=FOLDER)

# get validation data
dataset = fo.zoo.load_zoo_dataset("open-images-v7", split="validation", classes=CLASSES, max_samples=IMAGES, dataset_dir=FOLDER)
print(dataset)


#model = YOLO('yolov8n-seg.pt')
#results = model.train(data =
#'/Users/vhowle/Projects/image-segmentation-yolov8/config.yaml', epochs = 1, imgsz =640)

# This seems to be working. But very slow. Maybe try google collab as
# suggested in tutorial video? 
