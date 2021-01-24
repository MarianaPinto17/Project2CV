###########################################################################
# Mariana Pinto 84792 - OPENCV, PYTHON and DLIB                           #
# It loads face landmarks from a file, analyzes a webcam or an image,     #
# recognizes face, eyes, mouth and nouse and draw a square around the     #
# face on the image.                                                      #
###########################################################################

# HELP SOURCES: 
# http://dlib.net/
# To understand the use of facial landmarks: https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
# shape to np: https://stackoverflow.com/questions/58744107/how-to-get-the-coordinates-of-bounding-box-for-dets-in-dlib

############################## MODULES ####################################
import cv2
# if you don't have this library, install using pip install dlib (pip3 if using python3)
# this is a C++ library for extracting the facial landmarks
import dlib
import sys 
import numpy as np

####################### SHAPE TO NP #######################################
# function to convert dlib.full_object_detection to numpy array
# 68 landmarks to (x,y) coords
def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    #There are 68 landmark points on each face (2 DIMENSION)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

############################# MEMBERS DETECTION ###########################
def call(frame):
    # Convert image to grayscale to improve detection speed and accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    #Apply CLAHE to grayscale image
    clahe_image = clahe.apply(gray)

    #Detect the faces in the image
    face_detected = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    #For each detected face
    for x, y, w, h in face_detected:
        # draw it on the colour image "frame", with arguments: (coordinates), (size), (RGB color), line thickness 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        # creating the rectangle object from the outputs of haar cascade calssifier
        rect = dlib.rectangle(int(x),int(y),int(x+w),int(y+h))
        #Get coordinates
        shape = predictor(clahe_image, rect) 
        # using the first function to translate to dlib
        points = shape_to_np(shape)
        for i in points: 
            x = i[0]
            y = i[1]
            #For each point, draw a red circle with thickness2 on the original frame
            cv2.circle(frame, (x,y), 1, (0,255,0), 3) 

    #Display the frame
    cv2.imshow("image", frame) 
    
############################ INPUT ########################################
#sem imagem de entrada liga a camera
if len(sys.argv) == 1:
    capture = cv2.VideoCapture(0)
    is_image = False
#com imagem faz leitura da imagem
elif len(sys.argv) == 2:
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    is_image = True
else:
    print('Not possible, try again with "python3 members_recognition.py [file-name]"')

########################## LOAD CLASSIFIERS #############################
#load the trained classifier model
facecascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")

#landmark identifier
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

############################ SHOWS THE OBJECT ##########################
if is_image:
    frame = frame
    img = call(frame)
    cv2.waitKey(0)
else:
    while True:
        ret, frame = capture.read(); img = call(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break