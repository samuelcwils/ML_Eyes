import numpy as np
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import threading
import time

# Create a black canvas
canvas_size = (500, 500)  # Adjust size as needed
canvas = np.zeros(canvas_size, dtype=np.uint8)  # Black background

fig, (ax_canvas, ax_truth) = plt.subplots(1, 2, figsize=(10, 5))
im = ax_canvas.imshow(canvas, cmap='gray', vmin=0, vmax=255)
ax_canvas.set_title("Click and drag to draw")

# Load and display ground truth image
ground_truth_path = "Kvasir-SEG/masks/cju0qkwl35piu0993l0dewei2.jpg"
ground_truth = np.zeros(canvas_size, dtype=np.uint8)  # Placeholder for ground truth image
try:
    gt_image = Image.open(ground_truth_path).convert('L')
    gt_image = gt_image.resize(canvas_size)
    ground_truth = np.array(gt_image)
except Exception as e:
    print("Error loading ground truth image:", e)

gt_im = ax_truth.imshow(ground_truth, cmap='gray', vmin=0, vmax=255)
ax_truth.set_title("Ground Truth")

# Default brush size
brush_size = 20  # Increased brush size
drawing = False

def on_press(event):
    global drawing
    drawing = True
    draw(event)

def on_release(event):
    global drawing
    drawing = False

def draw(event):
    if event.xdata is not None and event.ydata is not None and drawing:
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:  # Left-click to draw
            value = 255
        elif event.button == 3:  # Right-click to erase
            value = 0
        else:
            return
        
        for i in range(-brush_size, brush_size + 1):
            for j in range(-brush_size, brush_size + 1):
                if 0 <= x + i < canvas_size[1] and 0 <= y + j < canvas_size[0]:
                    canvas[y + j, x + i] = value  # Set pixel value
        im.set_data(canvas)
        plt.draw()


def get_bwh_loss_fn(loss_type):
    if loss_type == "bwh_loss":
        def bwh_loss(y_true, y_pred):
    #approach to getting this to work. Do all non differentiable operations on truth matrix
    #also it might be better to do this computation outside the model, and just call it. 
            y_cast=tf.cast(y_true, tf.float32)
            centroid = tf.reduce_mean(tf.cast(tf.where(y_true>0),tf.float32), axis=0) #index of the centroid of the truth matrix

            #get the indicies of the in and out regions
            in_indicies = tf.where(y_true > 0)
            out_indicies = tf.where(y_true < 0.5)

            #get the area of regions
            area_in = tf.cast(tf.math.count_nonzero(y_true > 0.5) , tf.float32)
            area_out = tf.cast(tf.math.count_nonzero(y_true < 0.5) , tf.float32)

            weight_in = 2 - (tf.norm(tf.cast(in_indicies, tf.float32) - centroid, axis=1)) * (tf.sqrt(tf.constant(math.pi) / area_in))  #start punish in follows  2 - (normalized norm)
            weight_out = 1 + tf.norm(tf.cast(out_indicies, tf.float32) - centroid, axis=1) * (tf.sqrt(tf.constant(math.pi) / (area_out + area_in))) #start punish out follows 1 + (normalized norm)

            #update the truth matrix with the weights
            updated_truth_in_weight=tf.tensor_scatter_nd_update(y_cast, in_indicies, weight_in)
            weight_matrix=tf.tensor_scatter_nd_update(updated_truth_in_weight, out_indicies, weight_out)

            #compute loss
            elementwise_diff = y_true - y_pred
            weighted_loss=weight_matrix * elementwise_diff
            positive_loss=tf.abs(weighted_loss)

            #sums the values and takes mean
            loss = tf.math.reduce_mean(positive_loss)
            return loss
        return bwh_loss
    
    elif loss_type == "IoU_loss":
        def IoU_prototype_loss(y_true, y_pred):
            y_cast=tf.cast(y_true, tf.float32)
            centroid = tf.reduce_mean(tf.cast(tf.where(y_true>0),tf.float32), axis=0) #index of the centroid of the truth matrix

            #get the indicies of the in and out regions
            num_true = tf.cast(tf.math.count_nonzero(y_true) , tf.float32)
            num_pred = tf.cast(tf.math.count_nonzero(y_pred) , tf.float32)
            num_intersect = tf.cast(tf.math.count_nonzero(y_true * y_pred) , tf.float32)
            
            Jaccard_contribution = 1 - (num_intersect / (num_true + num_pred - num_intersect))  # Jaccard loss    
            #variation_contribution = tf.image.total_variation(y_pred)  # Total variation loss
            return 
        return IoU_prototype_loss
            

            
            



def compute_loss():
    global canvas
    ground_truth_norm = ground_truth / 255.0  # Normalize ground truth
    canvas_normalized = canvas / 255.0  # Normalize drawn canvas

    y_true = tf.convert_to_tensor(ground_truth_norm, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(canvas_normalized, dtype=tf.float32)

    loss_fn = get_bwh_loss_fn()
    loss = loss_fn(y_true, y_pred)
    print("Loss:", loss.numpy())

def loss_loop():
    while True:
        compute_loss()
        time.sleep(1/2)  # 60 FPS loss updates

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", draw)

# Start the loss computation thread
loss_thread = threading.Thread(target=loss_loop, daemon=True)
loss_thread.start()

plt.show()