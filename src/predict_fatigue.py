# Code pour la prédiction de la fatigue
import cv2
import numpy as np
import tensorflow as tf

def predict_fatigue(frame, model):
    resized_face = cv2.resize(frame, (128, 128))
    normalized_face = resized_face / 255.0
    reshaped_face = np.reshape(normalized_face, (1, 128, 128, 3))
    prediction = model.predict(reshaped_face)
    return prediction[0][0] > 0.5

if __name__ == "__main__":
    model = tf.keras.models.load_model("../models/fatigue_detection_model.h5")
    frame = cv2.imread("../data/test/sample.jpg")
    is_fatigued = predict_fatigue(frame, model)
    print("Fatigué" if is_fatigued else "Pas fatigué")