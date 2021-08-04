from keras.preprocessing.image import img_to_array
import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'C:\\Users\\nandi\\haarcascade_frontalface_default.xml'
emotion_model_path = 'C:\\Users\\nandi\\Downloads\\my_model.h5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised",
 "neutral"]


#feelings_faces = []
#for index, emotion in enumerate(EMOTIONS):
   # feelings_faces.append(cv2.imread('emojis/' + emotion + '.png', -1))

# starting video streaming
cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)
while True:
    frame = camera.read()[1]
    #reading the frame
    frame = imutils.resize(frame,width=300)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 28x28 pixels, and then prepare
            # the ROI for classification via the CNN
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
    else: continue

 
    for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                # construct the label text
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                # draw the label + probability bar on the canvas
               # emoji_face = feelings_faces[np.argmax(preds)]

                
                w = int(prob * 300)
                cv2.rectangle(canvas, (7, (i * 35) + 5),
                (w, (i * 35) + 35), (0, 0, 255), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 2)
                cv2.putText(frameClone, label, (fX, fY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),
                              (0, 0, 255), 2)
#    for c in range(0, 3):
#        frame[200:320, 10:130, c] = emoji_face[:, :, c] * \
#        (emoji_face[:, :, 3] / 255.0) + frame[200:320,
#        10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)


    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()































# import cv2
# import numpy as np
# import os
# import keras
# from keras.models import load_model


# dicts = ['empty', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# # To capture video from a webcam
# cap = cv2.VideoCapture(0)

# if not (cap.isOpened()):
#     print('Could not open video device')

# # Load the cascade
# face_cascade = cv2.CascadeClassifier('C:\\Users\\nandi\\haarcascade_frontalface_default.xml') 
# font = cv2.FONT_HERSHEY_SIMPLEX
# # To use a video file as input 
# # cap = cv2.VideoCapture('filename.mp4')
# model = keras.models.load_model('C:\\Users\\nandi\\Downloads\\my_model.h5', compile=False)
# counts = {}
# while True:
#     # Read the frame
#     _, frame = cap.read()
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Detect the faces
#     faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#     # Draw the rectangle around each face
#     for (x, y, w, h) in faces:
#         fc = frame[y:y+w, x:x+w]
#         # crop over resize
#         fin = cv2.resize(fc, (96, 96))
#         roi = cv2.resize(fc, (96, 96))
#         roi = np.expand_dims(roi, axis=0)
#         cropped_img_float = roi.astype(float)
#         pred = model.predict(cropped_img_float)
#         rounded_prediction = np.argmax(pred, axis=1)
#         emotion = dicts[rounded_prediction[0]]
#         cv2.putText(frame, str(emotion), (x, y), font, 1, (255, 255, 0), 2)
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
#     if cv2.waitKey(1) == 27:
#         break
#     cv2.imshow('Filter', frame)
#     # Stop if escape key is pressed
#     k = cv2.waitKey(30) & 0xff
#     if k==27:
#         break
# # Release the VideoCapture object
# cap.release()
