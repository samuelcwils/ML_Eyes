import keras
from keras import layers
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from dataset import get_tensorflow_dataset_split
from unet import build_unet_model
from neptunelogger import NeptunePredictionsLogger


def train(model, train_dataset, valid_dataset, test_dataset, seed, batch_size=48, learning_rate=0.001, n_filters=64, callbacks=[]):
    keras.utils.set_random_seed(seed) #make augmentation and loading the dataset consistent
    tf.config.experimental.enable_op_determinism()
    #Determinism breaks things for me. Has to do with the maxpool operation. I think it's CUDA. - Sam        
    
    model.compile(optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, epsilon=1e-04,),
                      loss = 'dice',
                      metrics=[keras.metrics.BinaryIoU(target_class_ids=[0, 1],threshold=0.5)])
                      #metrics=['accuracy', 'categorical_accuracy'])
                      #metrics=['accuracy', 'mse'])
    
    # Output the model summary. This shows the sizes of input and output at
    # each layer and number of parameters to be trained etc. 
    model.summary()
    
    # Train the model.
    model.fit(
        x=train_dataset,
        epochs=5,
        verbose="auto",
        callbacks=callbacks,
        validation_data=valid_dataset,
        shuffle=False,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
    )
    
    # check seed after
    #check_seed = torch.random.initial_seed()
    #print('check seed after code', check_seed)
    
    # Model evaluation (with the test data).
    score = model.evaluate(test_dataset, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
            
    # Model evaluation (with the validation data). This is used because of hyperparameter searching
    score = model.evaluate(valid_dataset, verbose=0)
    print("Validation loss:", score[0])
    print("Validation accuracy:", score[1])

    #optimize for accuracy
    return score[1]

def objective(trial, seed):
    tf.keras.backend.clear_session()
    
    batch_size = 16
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.001)
    n_filters = trial.suggest_int('n_filters', 64, 64)

    # Set path for dataset.
    # Assumes the following structure:
    #   data images in data_path/data/
    #   corresponding masks in data_path/labels/masks/0/
    
    # Old dataset. These were problematic due to mask and image sizes not
    # being the same. 
    #data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/small_train'
    #data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/Ducks/train'
    
    # This dataset works better and the images and masks have the same size.
    input_img_path = 'Kvasir-SEG/images/'
    mask_img_path = 'Kvasir-SEG/masks/'
    
    # For quick testing, I made a smaller subset of data.
    #data_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/images_small'
    #mask_path = '/Users/vhowle/Projects/ML_Eyes/DataImages/kvasir/Kvasir-SEG/masks_small'
    
    # For kvasir, images can be up to 256x256 to match smallest images.
    # Making smaller images for faster testing. 
    #min_height = 64
    #min_width = 64
    min_height = 256
    min_width = 256

    gpus = tf.config.list_logical_devices('GPU')
    strategy = tf.distribute.MirroredStrategy(gpus)
    with strategy.scope():
        mixed_precision.set_global_policy('mixed_float16')
    
        train_dataset, valid_dataset, test_dataset = get_tensorflow_dataset_split((min_width, min_height), input_img_path, mask_img_path, seed, batch_size, 0.8, 0.1, 0.1)
    
        model = build_unet_model(min_height, min_width, n_filters)

        #create a neptune instance for each trial
        trial_run = neptune.init_run(name="trial-" + str(trial.number), project='knightenjoyer15/Project', capture_hardware_metrics=True, api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZTE1ZGJhZi1jOTMyLTRiM2QtYTY3MC1jYzJlZTYyYTczODEifQ==") 
        predictions_neptune_callback = NeptunePredictionsLogger(trial_run, model, test_dataset) #log images of predictions to track model
        default_neptune_callback = NeptuneCallback(run=trial_run)
        
        accuracy = train(model, train_dataset, valid_dataset, test_dataset, seed, batch_size, learning_rate, n_filters, callbacks=[predictions_neptune_callback, default_neptune_callback])
        trial_run.stop()
        return accuracy