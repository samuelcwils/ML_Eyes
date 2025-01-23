import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from dataset import get_tensorflow_dataset

input_img_path = 'Kvasir-SEG/images/'
mask_img_path = 'Kvasir-SEG/masks/'

min_height = 256
min_width = 256


#this one displays a sample images, masks, and predictions 
def display_predictions(model, dataset, num_images=3):
    sample = next(iter(dataset))
    images = sample[0]
    masks = sample[1]
    predictions = model.predict(images)
    plt.figure(figsize=(15, num_images * 5))
    
    for i in range(num_images):
        # Original input image
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(np.array(images[i]))
        plt.title("Input Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(np.array(masks[i]).squeeze())
        plt.title("True Mask")
        plt.axis("off")

        # Predicted mask
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predictions[i].squeeze())
        plt.title("Predicted Mask")
        plt.axis("off")
        
    plt.show()

    #this one just shows images and masks
def display_dataset(dataset, num_images=3):
    sample = next(iter(dataset))
    images = sample[0]
    masks = sample[1]
    plt.figure(figsize=(15, num_images * 5))

    for i in range(num_images):
        # Original input image
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(np.array(images[i]))
        plt.title("Input Image")
        plt.axis("off")

        # Ground truth mask
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(np.array(masks[i]).squeeze())
        plt.title("True Mask")
        plt.axis("off")

        #Overlay
        im_color = np.stack((masks[i], masks[i], masks[i]), axis=-1)
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(np.array(im_color).squeeze() * np.array(images[i]))
        plt.title("Overlay")
        plt.axis("off")

        #Overlay for 3 channel mask
        # plt.subplot(num_images, 3, i * 3 + 3)
        # plt.imshow(np.array(masks[i]).squeeze() / 255.0 * np.array(images[i]))
        # plt.title("Overlay")
        # plt.axis("off")
        
    plt.show()

# unet_model = tf.keras.models.load_model('/home/samwilson/ML_Eyes-main/PythonExamples/models/model.keras')

# # Show the model architecture
# unet_model.summary()

seed = 500

dataset = get_tensorflow_dataset((min_width, min_height), input_img_path, mask_img_path, seed) 
dataset = dataset.batch(8)

display_dataset(dataset, num_images=8)