import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from keras.models import load_model
import matplotlib.pyplot as plt

# Load model
model = load_model("best_model.h5")

# Load Haar cascade
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, test_img = cap.read()  # Capture frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + h, x:x + w]  # Crop region of interest (face area) from image
        roi_gray = cv2.resize(roi_gray, (224, 224))  # Resize face image
        img_pil = Image.fromarray(roi_gray)  # Convert numpy array to PIL image
        img_array = np.array(img_pil)
        img_pixels = np.expand_dims(img_array, axis=0)
        img_pixels = img_pixels.astype('float32')
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # Find max index in predictions array
        max_index = np.argmax(predictions[0])

        emotions = ('angry: Punching bag workout,Anger journaling', 'disgust: Greative expression,environment cleanup', 'fear:Exposure therapy,mindfulness meditation', 'happy: gratitude practise,random act of kindness', 'sad: Emotional release through art,Reach out for support', 'surprise:Embrace spontaneity,Try something new', 'neutral:Explore new stuffs,Take a break')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    resized_img = cv2.resize(test_img, (500, 500))
    cv2.imshow('Facial Emotion Analysis', resized_img)

    if cv2.waitKey(10) == ord('q'):  # Wait until 'q' key is pressed
        break

cap.release()
cv2.destroyAllWindows()
