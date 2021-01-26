# Project2CV

The propose of this report has the primary objective of classifying facial expressions shown by a person for the  
Visual Computation project chosen by our group, using images or a webcam as an input. It is used OpenCV and Python 
to development.

The main goal of this project was to learn and create a program that detects facial expressions, mainly one of the
seven universal emotions – anger, contempt, disgust, fear, happiness, sadness and surprise. The implementation can
be categorized into four steps: face location, facial landmark location, extraction stage and emotion classification. 
It is used an appropriate database with several images that serves as training and testing our
model for each labeled emotion.

We use a small part of CK+ dataset which is labelled on 7 emotions, as stated above.
The dataset has 7 folders named with which emotion that will serve for test our program later.


To deploy our project, some prerequisites/libraries need to be installed:
        • OpenCV
        • Python language
        • Dlib (use pip 3 install dlib)
        • Sickit-learn (use pip 3 install sickit-learn)
        
We have 4 different types of programs:
        • Face_recognition.py: recognizes only faces
        • Members_recognition.py: recognizes facial landmarks and faces
        • Train_model.py: trains our model to recognise emotions
        • Emotions_recognition.py: recognizes faces, landmarks and emotions.
        
For running the APP you will need a terminal. You can run the APP using your webcam:

                python3 facial_recognition.py
                python3 members_recognition.py
                python3 emotions_recognition.py
                
Or you can run the APP using an image:
                
                python3 facial_recognition.py [image-path]
                
Finally, you can train our model using:

                 python3 train_model.py

