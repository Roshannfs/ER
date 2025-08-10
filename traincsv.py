import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class EmotionRecognitionModel:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.history = None

    def load_data(self):
        data = pd.read_csv(self.csv_path)
        pixels = data['pixels'].tolist()
        faces = []
        for pixel_sequence in pixels:
            face = np.array(pixel_sequence.split(), dtype='float32')
            face = face.reshape(48, 48, 1)
            faces.append(face)
        faces = np.array(faces) / 255.0
        emotions = to_categorical(data['emotion'], num_classes=7)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            faces, emotions, test_size=0.2, random_state=42
        )

    def build_model(self):
        self.model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(pool_size=(2,2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train(self, epochs=30, batch_size=64):
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size
        )

    def evaluate(self):
        loss, acc = self.model.evaluate(self.X_test, self.y_test)
        return loss, acc

    def save(self, path):
        self.model.save(path)

