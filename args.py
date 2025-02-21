import argparse

def get_args():

    #training args
    parser = argparse.ArgumentParser(description="Train image segmentation models with hyperparameter tuning and metadata logging")

    parser.add_argument("--seed", type=int, required=True, help="Seed for determinism")
    parser.add_argument("--use_optuna", action="store_true", help="Enable hyperparameter optimization with Optuna")
    parser.add_argument("--use_neptune", action="store_true", help="Enable metadata logging with Neptune")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, nargs="*", metavar=("MIN", "MAX"),
                        required=True, help="Learning rate value or range for optimization")
    parser.add_argument("--loss", type=str, nargs="*", choices=["dice", "binary_crossentropy"],
                        required=True, help="Loss function")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
    parser.add_argument("--n_trials", type=int, required=True, help="Number of trials for hyperparamter optimization")
    parser.add_argument("--use_mixed_precision", action="store_true", help="Enable mixed precision compute")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable the use of multiple gpus")
    parser.add_argument("--optimizer", nargs="*", type=str, choices=["adam", "sgd"],
                        required=True, help="Optimizer for training")
    parser.add_argument("--name", default=" ", type=str, help="The name of the study")
    parser.add_argument("--load_checkpoint", action="store_true", help="Name of file to load as a checkpoint for the optuna study")

    #model args
    parser.add_argument("--im_width", type=int, required=True, help="Image width")
    parser.add_argument("--im_height", type=int, required=True, help="Image height")
    parser.add_argument("--n_filters", type=int, nargs="*", metavar=("MIN", "MAX"),
                        required=True, help="Number of filters in the convolutional layers (single value or range)")
    parser.add_argument("--layer_depth", type=int, nargs="*", metavar=("MIN", "MAX"),
                        required=True, help="Depth of the network (single value or range)")
    parser.add_argument("--stack_num_down", type=int, nargs="*", metavar=("MIN", "MAX"),
                        required=True, help="Number of convolutional layers per downsampling block (single value or range)")
    parser.add_argument("--stack_num_up", type=int, nargs="*", metavar=("MIN", "MAX"),
                        required=True, help="Number of convolutional layers per upsampling block (single value or range)")
    parser.add_argument("--activation", type=str, nargs="*", choices=["ReLU", "Sigmoid", "tanh", "Softmax", "GELU", "selu", "swish"],
                        required=True, help="Activation function for hidden layers")
    parser.add_argument("--output_activation", type=str, nargs="*", choices=["Linear", "Softmax", "Sigmoid", "tanh"],
                        required=True, help="Activation function for the output layer")
    parser.add_argument("--batch_norm", action="store_true", help="Enable batch normalization")
    parser.add_argument("--model_type", type=str, choices=["unet_2d", "unet++"],
                        required=True, help="Type of model from keras-unet-collection")

    args = parser.parse_args()

    # replaces lists with single values to just the single values. Don't want to pass in a list when there should be an integer
    for key, value in vars(args).items():
        if isinstance(value, list) and len(value) and type(value[0] != 'string') == 1:
            setattr(args, key, value[0])  # Convert single-item list to its element

    # Convert Namespace object to dictionary
    args_dict = vars(args)

    # # remove args that are not set
    # pop_list = [] #list of values to remove
    # for key, value in args_dict.items():
    #     if value == None:
    #         pop_list.append(key)
    # for key in pop_list:
    #     args_dict.pop(key)
    
    #seperate arguments into those used for training and those used for models
    training_args = ["seed", "use_optuna", "use_neptune", "epochs", "learning_rate", "loss", "batch_size", "n_trials", "use_mixed_precision", "multi_gpu", "optimizer", "name", "load_checkpoint"]
    training_dict = {key: args_dict.pop(key) for key in training_args if key in args_dict}
    training_dict['im_width'] = args_dict['im_width']
    training_dict['im_height'] = args_dict['im_height'] #img size is both a training and model creation parameter
    model_dict = args_dict 

    print(training_dict)
    print(model_dict)

    # # Process arguments to return single values instead of lists when only one number is provided
    # for key in ["learning_rate", "n_filters", "layer_depth", "stack_num_down", "stack_num_up"]:
    #     if key in training_dict:
    #         if isinstance(training_dict[key], list) and len(training_dict[key]) == 1:
    #             training_dict[key] = training_dict[key][0]
    #     if key in model_dict:
    #         if isinstance(model_dict[key], list) and len(model_dict[key]) == 1:
    #             model_dict[key] = model_dict[key][0]

    return training_dict, model_dict