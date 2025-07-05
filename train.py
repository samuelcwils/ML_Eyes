"""This file trains the model.
It uses the keras library to train the model. It also uses the neptune library to log the training process and the predictions.
It uses the optuna library to optimize the hyperparameters of the model.
This is the bulk of the code."""
import keras
from keras import layers
import tensorflow as tf
from keras import mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import neptune
from neptune.integrations.tensorflow_keras import NeptuneCallback
from neptunelogger import NeptunePredictionsLogger
from keras_unet_collection import models
from functools import partial
import neptune.integrations.optuna as npt_utils
from dataset import get_tensorflow_dataset
import optuna
import os
from getmodel import getmodel
from args import get_args
import joblib
from getloss import getloss
from savecheckpoint import SaveCheckpoint
import gc
import math
#from customloss import get_bitmask_loss_fn

def train(model_args, im_width, im_height, use_neptune, use_mixed_precision, optimizer, loss, seed, epochs, batch_size, learning_rate, name, tags, use_optuna, subtrial=0, trial=None, high_punish=5, low_punish=1):
    keras.utils.set_random_seed(seed)
    
    if(use_mixed_precision):
        mixed_precision.set_global_policy('mixed_float16')

    model = getmodel(**model_args)
    optimizer = tf.keras.optimizers.get({"class_name": optimizer, "config": {"learning_rate": learning_rate}})

    loss_fn = getloss(loss)
    metric = keras.metrics.BinaryIoU(target_class_ids=[0, 1],threshold=0.5)
    outputs = model.output_names
    loss_dict = {name: loss_fn for name in model.output_names} #needed to provide a loss for each output when using a model that has multiple
    metrics_dict = {name: metric for name in model.output_names}

    model.compile(optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, amsgrad=True),
    loss=loss_dict,
    metrics=metrics_dict)
    model.summary()

    # This dataset works better and the images and masks have the same size.
    img_path = 'Kvasir-SEG/'
    
    height = im_width
    width = im_height

    train_dataset =  get_tensorflow_dataset((width, height), img_path + 'trainimages', img_path + 'trainmasks', seed, augmentation=True, batch = True, batch_size = batch_size, output_names=outputs)
    valid_dataset = get_tensorflow_dataset((width, height), img_path + 'validimages', img_path + 'validmasks', seed, augmentation=False, batch = True, batch_size = batch_size, output_names=outputs)
    eval_dataset = get_tensorflow_dataset((width, height), img_path + 'evalimages', img_path + 'evalmasks', seed, augmentation=False, batch = True, batch_size = batch_size, output_names=outputs)
    #train_dataset, valid_dataset, test_dataset = get_tensorflow_dataset_split((width, height), input_img_path, mask_img_path, seed, 0.8, 0.1, 0.1, batch_size, output_names=outputs)

    callbacks = []
    if(use_neptune):
        trial_run = neptune.init_run(name= (name + " standalone") if not use_optuna else (name + " " + "trial-" + str(trial.number) + "-" + str(subtrial)), project='knightenjoyer15/Project', capture_hardware_metrics=True, tags=tags) 
        #predictions_neptune_callback = NeptunePredictionsLogger(trial_run, model, valid_dataset) #log images of predictions to track model
        default_neptune_callback = NeptuneCallback(run=trial_run)

        if(trial is not None):
            trial_run["sys/group_tags"].add([tags[0] + "-" + name + " " + "trial-" + str(trial.number)])

        #callbacks = [predictions_neptune_callback, default_neptune_callback]
        callbacks.append(default_neptune_callback)
    
    # if(trial is not None):
    #     callbacks.append(optuna.integration.KerasPruningCallback(trial, "val_loss"))

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

    #no 32 bit out for mixed precision

    # Model evaluation (with the test data).

    score = model.evaluate(eval_dataset, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
            
    # Model evaluation (with the validation data). This is used because of hyperparameter searching
    score = model.evaluate(valid_dataset, verbose=0)
    print("Validation loss:", score[0])
    print("Validation accuracy:", score[1])

    #optimize for accuracy
    loss = score[0]

    if(use_neptune):
        trial_run.stop()
        del trial_run

    tf.keras.backend.clear_session()

    del model
    del train_dataset, valid_dataset, eval_dataset
    
    gc.collect()

    return loss

#function that wraps around the training loop. used for hyperparamter searching with optuna
def objective(trial, model_args, training_args, n_subtrials):

    #  # Make copies of the original argument dictionaries
    training_dict = training_args.copy()
    model_dict = model_args.copy()

    # Sample training hyperparameters and assign them back to the dict
    training_dict['learning_rate'] = trial.suggest_float(
        'learning_rate',
        training_dict['learning_rate'][0],
        training_dict['learning_rate'][1]
    )
    training_dict['loss'] = trial.suggest_categorical('loss', training_dict['loss'])
    training_dict['optimizer'] = trial.suggest_categorical('optimizer', training_dict['optimizer'])

    # Sample model hyperparameters and update the model dict directly
    model_dict['n_filters'] = trial.suggest_int('n_filters', model_dict['n_filters'][0], model_dict['n_filters'][1])
    model_dict['layer_depth'] = trial.suggest_int('layer_depth', model_dict['layer_depth'][0], model_dict['layer_depth'][1])
    model_dict['stack_num_down'] = trial.suggest_int('stack_num_down', model_dict['stack_num_down'][0], model_dict['stack_num_down'][1])
    model_dict['stack_num_up'] = trial.suggest_int('stack_num_up', model_dict['stack_num_up'][0], model_dict['stack_num_up'][1])
    model_dict['activation'] = trial.suggest_categorical('activation', model_dict['activation'])
    model_dict['output_activation'] = trial.suggest_categorical('output_activation', model_dict['output_activation'])

    if(task_index == 0):
        print(model_dict)
        print(training_dict)

    # Train model and return the evaluation metric
    #use the training_args dict again become some options are not selected by optuna (e.g. seed)

    values = []
    for i in range(n_subtrials):
        #num_invalid = 0
        value = train(subtrial=i, model_args=model_dict, trial=trial, **training_dict)

        while(math.isnan(value)):
            # num_invalid = num_invalid + 1
            # if(num_invalid == n_trials // 2):
            #     raise optuna.TrialPruned()
            value = train(subtrial=i, model_args=model_dict, trial=trial, **training_dict)
        training_dict["seed"] = training_dict["seed"] + 1
        values.append(value)
    mean_value = np.mean(values)

    tf.keras.backend.clear_session()
    gc.collect()

    return mean_value

if __name__=='__main__':
    
    task_index = int(os.environ.get('SLURM_PROCID', '0'))
    if(task_index == 0):
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    training_args, model_args = get_args()

    #some argparse options need to be taken out of the dictionaries
    use_optuna = training_args['use_optuna']
    n_trials = training_args.pop('n_trials')
    n_subtrials = training_args.pop('n_subtrials')
    load_checkpoint = training_args.pop("load_checkpoint")
    multi_gpu = training_args.pop('multi_gpu')
    dynamic_allocation = training_args.pop('dynamic_allocation')
    slurm = training_args.pop('slurm')
    tags = training_args['tags']
    name = training_args["name"]
    use_neptune = training_args['use_neptune']
    seed = training_args['seed']
    keras.utils.set_random_seed(seed) #make augmentation and loading the dataset consistent
    #tf.config.experimental.enable_op_determinism()

    project='knightenjoyer15/Project'
    api_key = os.environ.get('NEPTUNE_API_TOKEN')

    if(dynamic_allocation):
        gpus = tf.config.experimental.list_physical_devices('GPU') #normally tensorflow allocates all the memory on the gpu. This allows it to allocate only how much it needs. For debugging purposes.
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    #with hyperparameter optimization
    if(use_optuna):
        if(load_checkpoint):
            study = joblib.load(name + "_checkpoint.pkl")
        else:
            storage_name = "sqlite:///{}.db".format(name)
            study = optuna.create_study(study_name=name, direction='minimize', storage=storage_name, load_if_exists=True)
        objective_seeded = partial(objective, model_args=model_args, training_args=training_args, n_subtrials=n_subtrials) #need to pass study in so I can save it
        
        #create a neptune instance for the optuna study
        callbacks = []
        
        if(use_neptune and task_index == 0):
            study_run = neptune.init_run(name=name+" optuna-study", project=project, capture_hardware_metrics=True, api_token=api_key, tags=tags)
            neptune_callback = npt_utils.NeptuneCallback(study_run) #for logging metadata about hyperparamter optimization
            callbacks.append(neptune_callback)

        callbacks.append(SaveCheckpoint(name=name)) #add callback to save optuna study after every trial

        if(multi_gpu):
            # gpus = tf.config.list_logical_devices('GPU')
            # strategy = tf.distribute.MirroredStrategy(gpus)
            study.optimize(objective_seeded, callbacks=callbacks, n_trials=n_trials, gc_after_trial=True)#gc_after_trial enables garbage collection after each trial
        elif(slurm):
            # slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
            # communication = tf.distribute.experimental.CommunicationImplementation.NCCL
            # strategy = tf.distribute.MultiWorkerMirroredStrategy(
            #     cluster_resolver=slurm_resolver,
            #     communication_options=tf.distribute.experimental.CommunicationOptions(
            #         implementation=communication
            #     )
            # )
            # with strategy.scope():
                study.optimize(objective_seeded, callbacks=callbacks,  n_trials=n_trials, gc_after_trial=True)#gc_after_trial enables garbage collection after each trial
        else :
            #strategy = tf.distribute.get_strategy() 
            study.optimize(objective_seeded, callbacks=callbacks, n_trials=n_trials, gc_after_trial=True)#gc_after_trial enables garbage collection after each trial
        
        if(use_neptune and task_index == 0):
            study_run.stop()
    
    #no hyperparameter optimization
    else:
        if(multi_gpu):
            # gpus = tf.config.list_logical_devices('GPU')
            # strategy = tf.distribute.MirroredStrategy(gpus)
            train(model_args=model_args, trial_num=0,**training_args)
        elif(slurm):
            # slurm_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=15000)
            # communication = tf.distribute.experimental.CommunicationImplementation.NCCL
            # strategy = tf.distribute.MultiWorkerMirroredStrategy(
            #     cluster_resolver=slurm_resolver,
            #     communication_options=tf.distribute.experimental.CommunicationOptions(
            #         implementation=communication
            #     )
            # )
            train(model_args=model_args, trial_num=0,**training_args)
        else :
            #strategy = tf.distribute.get_strategy() 
            train(model_args=model_args,**training_args)