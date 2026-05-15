# Code pour entraîner un modèle CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_cnn_model(train_dir, val_dir, model_path):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
    val_generator = val_datagen.flow_from_directory(
        val_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=10, validation_data=val_generator)
    model.save(model_path)

    print(f"Modèle entraîné et sauvegardé à : {model_path}")

if __name__ == "__main__":
    train_cnn_model("../data/train", "../data/val", "../models/fatigue_detection_model.h5")