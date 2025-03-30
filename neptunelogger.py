import neptune
import tensorflow as tf
from neptune.types import File
import numpy as np

# Custom Callback for logging predictions (images) from the validation dataset for each epoch
class NeptunePredictionsLogger(tf.keras.callbacks.Callback):
    def __init__(self, run, model, valid_dataset):
        super().__init__()
        self.run = run
        self.n_outputs = len(model.output_names)
        self.valid_dataset = valid_dataset

    def on_epoch_end(self, batch, logs=None):
        valid_batch = next(iter(self.valid_dataset.take(1)))  # Get a validation batch
        valid_image = valid_batch[0]
        valid_mask = valid_batch[1]
        prediction = self.model.predict(valid_image)
        prediction = prediction[0][0] if self.n_outputs > 1 else prediction[0] #multiple mask outputs vs one. the last prediction is the 
        valid_image = valid_image[0].numpy()
        valid_mask = valid_mask.popitem()[1][0] if type(valid_mask) == dict else valid_mask[0].numpy() #if there are multiple masks grab the last one from the dictionary (they should all be the same)
        #pop the dictionary with masks then grab the tensor with masks and the first mask .pop() [0] [0]

        # Log image, ground truth mask, prediction, and overlay to Neptune
        mask_color = np.stack((valid_mask, valid_mask, valid_mask), axis=-1).squeeze()
        prediction_color = np.stack((prediction, prediction, prediction), axis=-1).squeeze() #convert from grayscale to color
        overlay = (np.array(prediction_color) * valid_image)
        overall_image_log = np.concatenate((valid_image, mask_color, prediction_color, overlay))
        self.run['valid_predictions/'].log(File.as_image(overall_image_log))