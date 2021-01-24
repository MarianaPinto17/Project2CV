###########################################################################
# Mariana Pinto 84792 - OPENCV and PYTHON                                 #
# It loads face recognition model from a file, shows a webcam or a image, #
# recognizes face and draw a square around the face on the image.         #
###########################################################################

#Help Source:


############################ MODULES ######################################
import cv2
import dlib
import pickle
import numpy as np
import math
import sys
from sklearn.svm import SVC

############################ INPUT ########################################
#if no arg turns on webcam
if len(sys.argv) == 1:
    video_capture = cv2.VideoCapture(0)
    is_image = False
# if there's an arg opens the image
elif len(sys.argv) == 2:
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    is_image = True
else:
    print('Not possible, try again with "python3 face_recognition.py [file-name]"')

############################## INITIALIZATION #############################
data = {} #Make dictionary for all values
data['landmarks_vectorised'] = [] #assign a key value to record landmarks
emotions = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

pkl_filename = 'pickle_model.pkl' #trained model file 
with open(pkl_filename, 'rb') as file:  #load all weights from model 
    pickle_model = pickle.load(file) 

detector = dlib.get_frontal_face_detector() #Face detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier.
FACE_SHAPE = (200, 200) #Size of capture frame - reAdjustable 

################################FACE LANDMARKS############################
def get_landmarks(image):
    detections = detector(image, 1)
    for k,d in enumerate(detections): #For all detected face instances individually
        shape = predictor(image, d) #Draw Facial Landmarks with the predictor class
        xlist = []
        ylist = []
        for i in range(1,68): #Store X and Y coordinates in two lists
            xlist.append(float(shape.part(i).x))
            ylist.append(float(shape.part(i).y))
        #record mean values of both X Y coordinates    
        xmean = np.mean(xlist)
        ymean = np.mean(ylist)
        #store central deviance 
        xcentral = [(x-xmean) for x in xlist]
        ycentral = [(y-ymean) for y in ylist]

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):#analysing presence of facial landmarks 
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            #extract center of gravity with mean of axis
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            #measuring distance and angle of each landmark from center of gravity 
            dist = np.linalg.norm(coornp-meannp)
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi))
        
        data['landmarks_vectorised'] = landmarks_vectorised#store landmarks in global dictionary 
    if len(detections) < 1: #if no landmarks were detected, store error in dictionary 
        data['landmarks_vestorised'] = "error"

############################### EMOTION RECOGNITION ################################
def call(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grayscale as our dataset was grayscale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) #Does Local adapative histogram equalization for improved feed 
    clahe_image = clahe.apply(gray) #applies LAHE 
    get_landmarks(clahe_image) #obtain landmarks from input feed 

    if data['landmarks_vectorised'] != "error": #if landmarks are detected..
        prediction_data = np.array(data['landmarks_vectorised']) #convert to numpy array ..
        predicted_labels = pickle_model.predict(prediction_data.reshape(1,-1)) #to get predicted values ...
        print (emotions[predicted_labels[0]]) #prints the predicted emotion   
    else:
         print("no face detected on this one")

    cv2.imshow("image", frame) #Display the webcam output

############################ SHOWS THE OBJECT ##########################
if is_image:
    frame = frame
    img = call(frame)
    cv2.waitKey(0)
else:
    while True:
        ret, frame = video_capture.read(); img = call(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break