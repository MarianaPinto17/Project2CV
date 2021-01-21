###########################################################################
# Mariana Pinto 84792 - OPENCV and PYTHON                                 #
# It loads face recognition model from a file, shows a webcam or a image, #
# recognizes face and draw a square around the face on the image.         #
###########################################################################

# HELP SOURCE: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection

############################ MODULES ######################################
import cv2
import numpy as np
import sys

#sem imagem de entrada liga a camera
if len(sys.argv) == 1:
    capture = cv2.VideoCapture(0)
    is_image = False
#com imagem faz leitura da imagem
elif len(sys.argv) == 2:
    frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
    is_image = True
else:
    print('Not possible, try again with "python3 face_recognition.py [file-name]"')

 #load the trained classifier model
facecascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")

while True:
    if is_image:
        frame = frame
    else:
        ret, frame = capture.read()
    
    # Convert image to grayscale to improve detection speed and accuracy
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    #Apply CLAHE to grayscale image
    clahe_image = clahe.apply(gray)

    #Run classifier on frame
    face_detected = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=20, minSize=(10, 10), flags=cv2.CASCADE_SCALE_IMAGE)

    #Draw rectangle around detected faces
    for (x, y, w, h) in face_detected:
        # draw it on the colour image "frame", with arguments: (coordinates), (size), (RGB color), line thickness 2
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    #display frame
    cv2.imshow("webcam", frame)
    #Exit program when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break