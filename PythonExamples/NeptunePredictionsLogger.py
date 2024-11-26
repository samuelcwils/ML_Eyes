import neptune
import tensorflow as tf
from neptune.types import File

# Custom Callback for logging test predictions for each batch
class NeptunePredictionsLogger(tf.keras.callbacks.Callback):
    def __init__(self, run, model, test_dataset):
        super().__init__()
        self.run = run
        self.test_dataset = test_dataset

    def on_batch_end(self, batch, logs=None):
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        print("Memory Info:", memory_info)
        test_image = self.test_dataset.take(5)  # Get a test batch
        predictions = self.model.predict(test_image)
        
        # Log predictions and labels to Neptune
        self.run[f'test_predictions/'].log(File.as_image(predictions[0]))