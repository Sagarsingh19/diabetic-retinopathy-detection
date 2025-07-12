import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import os

def generate_and_save_gradcam(model, img_array, last_conv_layer_name, save_path='output_image/gradcam.jpg'):
    # Create sub-model for Grad-CAM
    grad_model = Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Forward + gradient pass
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Prepare original image from img_array (convert to uint8)
    original_image = np.uint8(img_array[0] * 255.0)

    # Ensure it's 3-channel
    if original_image.shape[-1] == 1:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[-1] == 3:
        pass
    else:
        raise ValueError("Unexpected image shape: ", original_image.shape)

    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap_color, 0.4, 0)

    # Save result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, superimposed_img)

    return save_path, int(pred_index)

