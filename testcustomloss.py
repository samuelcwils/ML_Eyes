"""This file is used to test the custom loss function. 
It creates a canvas where the user can draw and see the loss value in real-time.
"""
import numpy as np
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

def get_bwh_loss_fn(high_punish, low_punish):
    def bwh_loss(y_true, y_pred):

        elementwise_diff = tf.cast(y_true - y_pred, tf.float32)
        big_loss_matrix = tf.cast(tf.where(elementwise_diff > 0), tf.float32)
        small_loss_matrix = tf.cast(tf.where(elementwise_diff < 0), tf.float32)
        centroid = tf.math.divide(tf.reduce_sum(big_loss_matrix, 0), tf.cast(tf.size(big_loss_matrix), tf.float32))
        big_loss = tf.norm(big_loss_matrix - centroid, 2)
        small_loss = tf.norm(small_loss_matrix - centroid, 2)
        total_loss = big_loss + small_loss
        return total_loss
    return bwh_loss

def compute_loss():
    global canvas
    ground_truth_norm = ground_truth / 255.0  # Normalize ground truth
    canvas_normalized = canvas / 255.0  # Normalize drawn canvas

    y_true = tf.convert_to_tensor(ground_truth_norm, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(canvas_normalized, dtype=tf.float32)

    loss_fn = get_bwh_loss_fn(high_punish=2.0, low_punish=1.0)
    loss = loss_fn(y_true, y_pred)
    print("Loss:", loss.numpy())

def loss_loop():
    while True:
        compute_loss()
        time.sleep(1/60)  # 60 FPS loss updates

fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("button_release_event", on_release)
fig.canvas.mpl_connect("motion_notify_event", draw)

# Start the loss computation thread
loss_thread = threading.Thread(target=loss_loop, daemon=True)
loss_thread.start()

plt.show()