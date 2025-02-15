import keras
from keras import layers
import tensorflow as tf
from keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from dataset import get_tensorflow_dataset_split
from unet import build_unet_model
from neptunelogger import NeptunePredictionsLogger
from keras_unet_collection import models
from functools import partial
import neptune.integrations.optuna as npt_utils
import optuna
import os
from getmodel import getmodel
from args import get_args
from getloss import getloss

def train(model, trial_num, im_width, im_height, use_neptune, use_mixed_precision, optimizer, loss, seed, epochs, batch_size, learning_rate, name):
    tf.keras.backend.clear_session()
    keras.utils.set_random_seed(seed) #make augmentation and loading the dataset consistent
    tf.config.experimental.enable_op_determinism()

    
    # This dataset works better and the images and masks have the same size.
    input_img_path = 'Kvasir-SEG/images/'
    mask_img_path = 'Kvasir-SEG/masks/'
    
    height = im_width
    width = im_height

    train_dataset, valid_dataset, test_dataset = get_tensorflow_dataset_split((width, height), input_img_path, mask_img_path, seed, 0.8, 0.1, 0.1, batch_size)

    print("train_dataset size with batches: " + str(train_dataset.cardinality()))

    #no 32 bit out for mixed precision

    if(use_mixed_precision):
        mixed_precision.set_global_policy('mixed_float16')

    #create a neptune instance for each trial
    if(use_neptune):
        trial_run = neptune.init_run(name=name + " " + "trial-" + str(trial_num), project='knightenjoyer15/Project', capture_hardware_metrics=True) 
        predictions_neptune_callback = NeptunePredictionsLogger(trial_run, model, valid_dataset) #log images of predictions to track model
        default_neptune_callback = NeptuneCallback(run=trial_run)

        callbacks = [predictions_neptune_callback, default_neptune_callback]
    else:
        callbacks = []

    optimizer = tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}})

    model.compile(optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, epsilon=1e-04,),
                      loss = getloss(loss, model.outputs),
                      metrics=[keras.metrics.BinaryIoU(target_class_ids=[0, 1],threshold=0.5)])
                      #metrics=['accuracy', 'categorical_accuracy'])
                      #metrics=['accuracy', 'mse'])
    
    # Output the model summary. This shows the sizes of input and output at
    # each layer and number of parameters to be trained etc. 
    model.summary()
    
    # Train the model.
    model.fit(
        x=train_dataset,
        epochs=epochs,
        verbose="auto",
        callbacks=callbacks,
        validation_data=valid_dataset,
        batch_size=None,
        shuffle=False,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_freq=1,
    )
    
    # Model evaluation (with the test data).
    score = model.evaluate(test_dataset, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
            
    # Model evaluation (with the validation data). This is used because of hyperparameter searching
    score = model.evaluate(valid_dataset, verbose=0)
    print("Validation loss:", score[0])
    print("Validation accuracy:", score[1])

    #optimize for accuracy
    accuracy = score[1]

    if(use_neptune):
        trial_run.stop()

    return accuracy

#function that wraps around the training loop. used for hyperparamter searching with optuna
def objective(trial, model_args, training_args):

    #maybe this is a bad solution but because im popping variables I need to make a copy of the arg dicts so each iteration they can be used again
    training_dict = training_args.copy()
    model_dict = model_args.copy()

    # Extract hyperparameter ranges
    learning_rate_range = training_dict.pop('learning_rate')
    loss_types = training_dict.pop('loss')
    optimizer_types = training_dict.pop('optimizer')

    n_filter_range = model_dict.pop('n_filters')
    layer_depth_range = model_dict.pop('layer_depth')
    stack_num_down_range = model_dict.pop('stack_num_down')
    stack_num_up_range = model_dict.pop('stack_num_up')
    activation_types = model_dict.pop('activation')
    output_activation_types = model_dict.pop('output_activation')

    # Sample hyperparameters
    learning_rate = trial.suggest_float('learning_rate', learning_rate_range[0], learning_rate_range[1])
    loss = trial.suggest_categorical('loss', loss_types)
    optimizer = trial.suggest_categorical('optimizer', optimizer_types)

    n_filters = trial.suggest_int('n_filters', n_filter_range[0], n_filter_range[1])
    layer_depth = trial.suggest_int('layer_depth', layer_depth_range[0], layer_depth_range[1])
    stack_num_down = trial.suggest_int('stack_num_down', stack_num_down_range[0], stack_num_down_range[1])
    stack_num_up = trial.suggest_int('stack_num_up', stack_num_up_range[0], stack_num_up_range[1])
    activation = trial.suggest_categorical('activation', activation_types)
    output_activation = trial.suggest_categorical('output_activation', output_activation_types)

    # Initialize model with sampled hyperparameters
    model = getmodel(
        n_filters=n_filters,
        layer_depth=layer_depth,
        stack_num_down=stack_num_down,
        stack_num_up=stack_num_up,
        activation=activation,
        output_activation=output_activation,
        **model_dict  # Pass any remaining arguments
    )

    # Train model and return the evaluation metric
    #use the training_args dict again become some options are not selected by optuna (e.g. seed)
    return train(model, trial.number, learning_rate=learning_rate, loss=loss, optimizer=optimizer, **training_dict) 

if __name__=='__main__':

    training_args, model_args = get_args()

    #some argparse options need to be taken out of the dictionaries
    n_trials = training_args.pop('n_trials')
    use_optuna = training_args.pop('use_optuna')
    multi_gpu= training_args.pop('multi_gpu')
    name = training_args["name"]
    use_neptune = training_args['use_neptune']
    use_mixed_precision = training_args['use_mixed_precision']

    project='knightenjoyer15/Project'
    api_key = os.environ.get('NEPTUNE_API_TOKEN')

    #with hyperparameter optimization
    if(use_optuna):
        
        study = optuna.create_study(direction='maximize')
        objective_seeded = partial(objective, model_args=model_args, training_args=training_args) #input seed for determinism
        
        #create a neptune instance for the optuna study
        callbacks = []
        if(use_neptune):
            study_run = neptune.init_run(name=name+" optuna-study", project=project, capture_hardware_metrics=True, api_token=api_key)
            neptune_callback = npt_utils.NeptuneCallback(study_run) #for logging metadata about hyperparamter optimization
            callbacks = [neptune_callback]

        if(multi_gpu):
            gpus = tf.config.list_logical_devices('GPU')
            strategy = tf.distribute.MirroredStrategy(gpus)
            with strategy.scope():
                study.optimize(objective_seeded, callbacks=callbacks,  n_trials=n_trials, gc_after_trial=True)#gc_after_trial enables garbage collection after each trial
        else :
            study.optimize(objective_seeded, callbacks=callbacks, n_trials=n_trials, gc_after_trial=True)#gc_after_trial enables garbage collection after each trial
        
        if(use_neptune):
            study_run.stop()
    
    #no hyperparameter optimization
    else:
        model = getmodel(**model_args)
        if(multi_gpu):
            gpus = tf.config.list_logical_devices('GPU')
            strategy = tf.distribute.MirroredStrategy(gpus)
            with strategy.scope():
                train(model, 0, use_neptune, use_mixed_precision, **training_args)
        else :
            train(model, 0, use_neptune, use_mixed_precision, **training_args)