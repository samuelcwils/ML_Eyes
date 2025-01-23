import neptune
import tensorflow as tf
from neptune.types import File
import numpy as np

# Custom Callback for logging test predictions (images) for each batch
class NeptunePredictionsLogger(tf.keras.callbacks.Callback):
    def __init__(self, run, model, test_dataset):
        super().__init__()
        self.run = run
        self.test_dataset = test_dataset

    def on_epoch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print("Memory Info:", memory_info)
            
        test_batch = next(iter(self.test_dataset.take(1)))  # Get a test batch
        test_image = test_batch[0]
        test_mask = test_batch[1]
        prediction = self.model.predict(test_image)
        test_image = test_image.numpy().squeeze()
        test_mask = test_mask.numpy().squeeze()
        
        # Log image, ground truth mask, prediction, and overlay to Neptune
        mask_color = np.stack((test_mask, test_mask, test_mask), axis=-1).squeeze()
        prediction_color = np.stack((prediction, prediction, prediction), axis=-1).squeeze() #convert from grayscale to color
        overlay = (np.array(prediction_color) * test_image)
        overall_image_log = np.concatenate((test_image, mask_color, prediction_color, overlay))
        self.run['test_predictions/'].log(File.as_image(overall_image_log))