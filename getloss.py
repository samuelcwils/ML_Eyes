import keras

def getloss(loss_name, outputs):
    if(loss_name == 'custom'):
        return None #not implemented yet
    else:
        return keras.losses.get(loss_name)