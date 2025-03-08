import cv2
import numpy as np

import cv2
import numpy as np

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))  # Resize to match model input size
    img = img.astype("float32") / 255.0  # Normalize pixel values

    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    img = np.expand_dims(img, axis=0)   # Add batch dimension

    print("Processed Image Shape:", img.shape)  # Debugging
    return img



def decode_predictions(predictions):
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    
    predicted_indices = np.argmax(predictions, axis=-1)
    print("Predicted Indices:", predicted_indices)  # Debugging

    if predicted_indices.ndim == 1:
        predicted_text = "".join([characters[i] for i in predicted_indices])
    else:
        predicted_text = "".join([characters[i] for i in predicted_indices.flatten()])

    print("Decoded Text:", predicted_text)  # Debugging
    return predicted_text
