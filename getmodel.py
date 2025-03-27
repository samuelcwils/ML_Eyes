from keras_unet_collection import models

def getmodel(**kwargs):
    """Creates a Segmentation model using the specified parameters."""
    
    # Extracting required values from kwargs
    model_type = kwargs.pop("model_type")
    im_width, im_height = kwargs.pop("im_width"), kwargs.pop("im_height")
    n_filters, layer_depth = kwargs.pop("n_filters"), kwargs.pop("layer_depth")
    
    # a list that defines the number of convolutional filters per down- and up-sampling blocks.
    filter_num = [n_filters * 2**i for i in range(layer_depth)]
    if model_type == "unet_2d":
        return models.unet_2d(
            input_size=(im_width, im_height, 3), n_labels=1, pool="max", unpool='nearest', filter_num=filter_num, **kwargs, name="unet"
        )

    if model_type == "xnet":
        return models.unet_plus_2d(
            input_size=(im_width, im_height, 3), filter_num=filter_num, n_labels=1, **kwargs, pool='max', deep_supervision=True, unpool=False, name='xnet'
            )

    if model_type == "unet3plus":
        return models.unet_3plus_2d(input_size=(im_width, im_height, 3), filter_num_down=filter_num, filter_num_skip='auto', filter_num_aggregate='auto', n_labels=1, **kwargs, pool='max', unpool=False, 
                deep_supervision=True, name='unet3plus'
            )

    if model_type == "attunet":
        return models.att_unet_2d(
            input_size=(im_width, im_height, 3), filter_num=filter_num, n_labels=1, **kwargs, atten_activation='ReLU', attention='add', #output_activation=None, 
            pool=False, unpool='bilinear', name='attunet'
            )