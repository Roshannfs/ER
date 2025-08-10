import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_emotion_model(model_path):
    return load_model(model_path)

def get_face_detector():
    return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_roi):
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = face_roi.astype('float32') / 255.0
    face_roi = np.expand_dims(face_roi, axis=0)
    face_roi = np.expand_dims(face_roi, axis=-1)
    return face_roi

def predict_emotion(model, face_roi, emotion_labels):
    prediction = model.predict(face_roi)
    max_index = np.argmax(prediction)
    return emotion_labels[max_index]

def run_emotion_recognition(model_path, emotion_labels):
    model = load_emotion_model(model_path)
    face_cascade = get_face_detector()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]
            face_roi_processed = preprocess_face(face_roi)
            emotion = predict_emotion(model, face_roi_processed, emotion_labels)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_PATH = r'B:\git\ER\emotion_recognition_model.h5'
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    run_emotion_recognition(MODEL_PATH, EMOTION_LABELS)
