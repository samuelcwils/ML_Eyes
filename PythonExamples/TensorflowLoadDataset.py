import keras
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
from tensorflow import dtypes
import os
import tensorflow as tf
import albumentations as A
import cv2

def get_tensorflow_dataset(
    img_size,
    input_img_path,
    mask_img_path,

):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, mask_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, dtype="uint8", channels=3)
        input_img = tf_image.resize(input_img, img_size, method="nearest")
        input_img = tf_image.convert_image_dtype(input_img, "float64") 
        # I convert to floating point in the original code images are floating point. A different datatype may be better for the masks because of the loss function.

        mask_img = tf_io.read_file(mask_img_path)
        mask_img = tf_io.decode_png(mask_img, dtype = "uint8", channels=3)
        mask_img = tf_image.resize(mask_img, img_size, method="nearest")
        #mask_img = tf.where(mask_img > 10, 1, 0) 
        #This can set all values greater than a certain threshold to a certain value. Could make highlighted pixels 1 and others 0. Useful for some loss functions.
        mask_img = tf.cast(mask_img, dtype=tf.float64)

        return input_img, mask_img
        
    def augment_data(image, mask):
        transforms = A.Compose([
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(),
            ], additional_targets={'mask': 'image'})
    
        def aug_fn(image, mask):
            data = {"image":image, "mask":mask}
            aug_data = transforms(**data)
            aug_img = aug_data["image"]
            aug_mask = aug_data["mask"]
        
            return aug_img, aug_mask
        
        #to use albumentations I need to use this function then convert back to tensors
        aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout=[tf.float64, tf.float64])
        aug_img = tf.convert_to_tensor(aug_img)
        aug_mask = tf.convert_to_tensor(aug_mask)
        aug_img = tf.ensure_shape(aug_img, (*img_size, 3))
        aug_mask = tf.ensure_shape(aug_mask, (*img_size, 3))
        #Somehow the tensors lose shape after augmentation but I just set them back. Causes an error otherwise

        return aug_img, aug_mask

    input_img_paths = [os.path.join(input_img_path, file) for file in os.listdir(input_img_path) if os.path.isfile(os.path.join(input_img_path, file))]
    mask_img_paths = [os.path.join(mask_img_path, file) for file in os.listdir(mask_img_path) if os.path.isfile(os.path.join(mask_img_path, file))]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, mask_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE) #the function put into .map is applied dynamically when a batch is requested
    dataset = dataset.map(augment_data, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset
