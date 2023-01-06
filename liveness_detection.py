from imutils.video import VideoStream
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2 as cv
import os
from rich.console import Console

def detector(face,liveness_model,le):
    
            face = cv.resize(face, (32, 32))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            preds = liveness_model.predict(face)[0]
            
            j = np.argmax(preds)
            label = le.classes_[j]
            if label == 'real':
                return(True)
            else:
                return(False)
            #print(preds)
            #print(j)
            #print(label)

            label = "{}: {:.4f}".format(label, preds[j])
            
            if preds[j] > 0.50 and j == 1:
                cv.rectangle(img, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
                _label = "Liveness: {:.4f}".format(preds[j])
                cv.putText(img, _label, (startX, startY - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv.rectangle(img, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
                _label = "Fake: {:.4f}".format(preds[j])
                cv.putText(img, _label, (startX, startY - 10),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

