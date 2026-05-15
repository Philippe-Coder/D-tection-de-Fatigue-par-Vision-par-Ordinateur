# Fonctions utilitaires
import cv2
import numpy as np

def preprocess_image(image):
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    return np.reshape(normalized_image, (1, 128, 128, 3))