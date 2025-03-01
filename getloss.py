import tensorflow as tf
import keras

#gave defaults for punishing values so you don't have to enter them when not using the custom loss. Not sure if this is the best
#way to do this. Ian let me know what you think.
def getloss(loss_name, high_punish = 1.3, low_punish = 1):
    if(loss_name == 'custom_new'):
        def bwh_loss(y_true, y_pred):

            elementwise_diff = tf.cast(y_true - y_pred, tf.float32)
            big_loss_matrix = tf.cast(tf.where(elementwise_diff > 0), tf.float32)
            small_loss_matrix = tf.cast(tf.where(elementwise_diff < 0), tf.float32)
            centroid = tf.math.divide(tf.reduce_sum(big_loss_matrix, 0), tf.cast(tf.size(big_loss_matrix), tf.float32))
            big_loss = tf.norm(big_loss_matrix - centroid, 2, axis = 0)
            small_loss = tf.norm(small_loss_matrix - centroid, 2, axis = 0)
            total_loss = big_loss + small_loss
            return total_loss
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