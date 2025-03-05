"""This file contains the custom loss functions used in the model. 
The custom loss functions are used to train the model and are passed to the model during compilation."""
import tensorflow as tf
import keras

#gave defaults for punishing values so you don't have to enter them when not using the custom loss. Not sure if this is the best
#way to do this. Ian let me know what you think.
def getloss(loss_name, high_punish = 1.3, low_punish = 1):
    if(loss_name == 'custom_new'):
        def bwh_loss(y_true, y_pred):
#approach to getting this to work. Do all non differentiable operations on truth matrix
#also it might be better to do this computation outside the model, and just call it. 
            y_cast=tf.cast(y_true, tf.float32)
            centroid = tf.reduce_mean(tf.cast(tf.where(y_true>0),tf.float32), axis=0)
            in_indicies = tf.where(y_true > 0)
            out_indicies = tf.where(y_true < 0.5)
            weight_in = tf.norm(tf.cast(in_indicies, tf.float32) - centroid, axis=1)
            weight_out = tf.norm(tf.cast(out_indicies, tf.float32) - centroid, axis=1)
            updated_truth_in_weight=tf.tensor_scatter_nd_update(y_cast, in_indicies, weight_in)
            weight_matrix=tf.tensor_scatter_nd_update(updated_truth_in_weight, out_indicies, weight_out)
            
            elementwise_diff = y_true - y_pred
            weighted_loss=weight_matrix * elementwise_diff
            positive_loss=tf.abs(weighted_loss)
            #sums the values and takes mean
            loss = tf.math.reduce_mean(positive_loss)
            return loss
        return bwh_loss
    elif(loss_name == "custom_old"):
        def bitmask_loss_fn(y_true, y_pred):
            #truth matrix created
            #+1 if missed truth
            #-1 if predicted wrong
            elementwise_diff = y_true - y_pred

            #applies a if element<0 then high punishment else low punishment 
            weight_matrix = tf.where(elementwise_diff < 0, high_punish , low_punish)
            #element by element multiplication
            #could also be written as tf.math.multiply(weight_matrix * elementwise_loss)
            weighted_loss=weight_matrix * elementwise_diff
            #rturns positive values of the loss
            positive_loss=tf.abs(weighted_loss)
            #sums the values and takes mean
            loss = tf.math.reduce_mean(positive_loss)
            return loss
        return bitmask_loss_fn
    else:
        return keras.losses.get(loss_name)