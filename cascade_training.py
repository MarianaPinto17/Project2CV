# HELP SOURCE : https://docs.opencv.org/2.4/doc/user_guide/ug_traincascade.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_objdetect/py_face_detection/py_face_detection.html#face-detection
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
# http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/ --> adapted


##########MODULES####################
import cv2
# if you don't have this library, install using pip install dlib (pip3 if using python3)
import dlib
import sys 
import numpy as np

##### FEED CAMERA ##### 
# TO USE WEB CAMERA UNCOMMNET THIS LINE
capture = cv2.VideoCapture(0)

################IMAGE OPEN###################

#frame = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)

#if np.shape(frame) == ():
#    # Failed Reading
#    print("Image file could not be open!")
#    exit(-1)

#para detetar caras
detector = dlib.get_frontal_face_detector()

#landmark identifier
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    #TO USE WEB CAMERA UNCOMMNET THIS LINE
    ret, frame = capture.read()

    #passar para grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)
    #detetar caras na imagem
    detections = detector(clahe_image,1)
    for k,d in enumerate(detections):
        shape = predictor(clahe_image,d)
        for i in range(1,68):
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) #For each point, draw a red circle with thickness2 on the original frame
    cv2.imshow("image", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break