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
    seed,
    shuffle_buffer_fraction=0.1
):
    """Returns a TF Dataset."""

    def load_imgs(input_img_path, mask_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, dtype="uint8", channels=3)
        input_img = tf_image.resize(input_img, img_size, method="nearest")
        input_img = tf_image.convert_image_dtype(input_img, "float64") 
        # I convert to floating point in the original code images are floating point. A different datatype may be better for the masks because of the loss function.

        mask_img = tf_io.read_file(mask_img_path)
        mask_img = tf_io.decode_png(mask_img, dtype = "uint8", channels=1)
        mask_img = tf_image.resize(mask_img, img_size, method="nearest")
        
        mask_img = tf.where(mask_img > 10, 1, 0) 
        #This can set all values greater than a certain threshold to a certain value. Could make highlited pixels 1 and others 0. Useful for some loss functions.
        
        mask_img = tf.cast(mask_img, dtype=tf.float64)

        return input_img, mask_img
        
    def augment_data(image, mask):
        transforms = A.Compose([
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(),
            ],additional_targets={'mask': 'image'})
    
        def aug_fn(image, mask):
            data = {"image":image, "mask":mask}
            aug_data = transforms(**data)
            aug_img = aug_data["image"]
            aug_mask = aug_data["mask"]
        
            return aug_img, aug_mask

        aug_img, aug_mask = tf.numpy_function(func=aug_fn, inp=[image, mask], Tout=[tf.float64, tf.float64])
        aug_img = tf.convert_to_tensor(aug_img)
        aug_mask = tf.convert_to_tensor(aug_mask)
        aug_img = tf.ensure_shape(aug_img, (*img_size, 3))
        aug_mask = tf.ensure_shape(aug_mask, (*img_size, 1))

        return aug_img, aug_mask

    keras.utils.set_random_seed(seed) #make augmentation and loading the dataset consistent
    tf.config.experimental.enable_op_determinism()

    input_img_paths = [os.path.join(input_img_path, file) for file in os.listdir(input_img_path) if os.path.isfile(os.path.join(input_img_path, file))]
    mask_img_paths = [os.path.join(mask_img_path, file) for file in os.listdir(mask_img_path) if os.path.isfile(os.path.join(mask_img_path, file))]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, mask_img_paths))
    dataset = dataset.map(load_imgs, num_parallel_calls=tf_data.AUTOTUNE) #the function put into .map is applied dynamically when a batch is requested
    dataset = dataset.map(augment_data, num_parallel_calls=tf_data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=int(int(dataset.cardinality()) * shuffle_buffer_fraction) , reshuffle_each_iteration=True)

    return dataset


def get_tensorflow_dataset_split(img_size, input_img_path, mask_img_path, seed, batch_size, train_fraction, valid_fraction, test_fraction):
    
    if(train_fraction + valid_fraction + test_fraction != 1):
        raise Exception("Dataset fractions do not add up to one!")
        
    train_batch_size = batch_size
    valid_batch_size = 16
    test_batch_size = 1
    
    dataset = get_tensorflow_dataset(img_size, input_img_path, mask_img_path, seed) 
    #dataset = dataset.shuffle(buffer_size=int(int(dataset.cardinality()) * shuffle_buffer_fraction) , reshuffle_each_iteration=True)

    train_dataset = dataset.take(int(len(dataset) * train_fraction)) #allot samples to train
    temp_dataset = dataset.skip(int(len(dataset) * train_fraction)) #allot rest to other datasets. store the non train samples in a temp variable
    valid_dataset = temp_dataset.take(int(len(dataset) * valid_fraction)) #take the samples needed for valid
    test_dataset = temp_dataset.skip(int(len(dataset) * valid_fraction)) #the rest go to test

    print("train_dataset size: " + str(train_dataset.cardinality()))
    print("valid_dataset size: " + str(valid_dataset.cardinality()))
    print("test_dataset size: " + str(test_dataset.cardinality()))
    
    train_dataset = train_dataset.batch(train_batch_size).prefetch(tf.data.AUTOTUNE)
    valid_dataset = valid_dataset.batch(valid_batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(test_batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, valid_dataset, test_dataset