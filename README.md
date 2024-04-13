# EmoSuggest: Mood-Based Activity Suggestion 

EmotiGuide is an innovative project aimed at suggesting activities based on the user's mood, utilizing image emotion detection techniques. By leveraging machine learning and deep learning models, EmotiGuide can analyze images to determine the user's emotional state and recommend relevant activities tailored to their mood.
## Features
Image Emotion Detection: Utilizes deep learning models like MobileNet to detect emotions from user-provided images.
Activity Suggestion: Recommends activities based on the detected emotion, providing personalized suggestions for the user.

## Technologies Used
Python Libraries:
    NumPy
    Pandas
    Matplotlib
Deep Learning Framework:
    Keras
Pre-trained Models:
    MobileNet for image emotion detection
Image Data Processing:
    ImageDataGenerator for preprocessing images
Loss Function:
    Categorical Crossentropy for model training

## How It Works
Image Input: Users upload or provide images representing their current mood.
Emotion Detection: EmoSuggest utilizes MobileNet to analyze the images and detect the predominant emotion.
Activity Recommendation: Based on the detected emotion, EmotiGuide suggests relevant activities from a predefined set.


## LIVE VIDEO FACIAL EMOTION ANALYSIS AND ACTIVITY SUGGESTION
It also utilizes live video feed from a webcam to perform facial emotion analysis in real-time and suggest relevant activities based on the detected emotions. It employs machine learning techniques to recognize facial expressions and recommend activities tailored to the user's mood.
## Technologies Used
OpenCV: Used for capturing video from the webcam and performing real-time face detection.
TensorFlow and Keras: Utilized to load a pre-trained deep learning model for facial emotion recognition.
NumPy: Employed for array manipulation and data processing.
PIL (Python Imaging Library): Utilized to convert image arrays to PIL images for compatibility with the pre-trained model.
Matplotlib: Used for displaying the output with annotated facial emotion and suggested activities.

## How It Works
Capture Video: The script captures video frames from the webcam in real-time.
Face Detection: OpenCV's Haar Cascade Classifier is used to detect faces in the video frames.
Emotion Recognition: For each detected face, the script performs facial emotion recognition using a pre-trained deep learning model loaded with TensorFlow and Keras.
Activity Suggestion: Based on the detected emotion, the script suggests relevant activities from a predefined set.
Display Output: The video frames are displayed with bounding boxes around detected faces and annotations indicating the predicted emotion and suggested activity.

## Pre-trained Model
The deep learning model used for facial emotion recognition is loaded from a pre-trained Keras model file named best_model.h5.
Activities Based on Emotions

## The suggested activities for each detected emotion are as follows:
Angry: Punching bag workout, Anger journaling
Disgust: Creative expression, Environment cleanup
Fear: Exposure therapy, Mindfulness meditation
Happy: Gratitude practice, Random act of kindness
Sad: Emotional release through art, Reach out for support
Surprise: Embrace spontaneity, Try something new
Neutral: Explore new stuff, Take a break
![image](https://github.com/sarahasan17/EmoSuggest/assets/103211125/65005d98-741a-49a5-89f8-77bbe6d2dda2)

![image](https://github.com/sarahasan17/EmoSuggest/assets/103211125/cf7043da-0ed4-45d7-9a0d-0a31049d871c)



## Getting Started
Clone the Repository: Clone this repository to your local machine.
Install Dependencies: Install the required dependencies using pip.
Run the Application: Execute the main script to launch the EmotiGuide application.

