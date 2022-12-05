from imutils.video import VideoStream
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import numpy as np
import imutils
import pickle
import time
import cv2 as cv
import os

def detector():
    print("[INFO] loading face detector...")
    protoPath = "./face_detector//deploy.prototxt"
    modelPath = "./face_detector//res10_300x300_ssd_iter_140000.caffemodel"
    net = cv.dnn.readNetFromCaffe(protoPath, modelPath)

    print("[INFO] loading liveness detector...")
    model = load_model('liveness_detector_model')
    le = pickle.loads(open("le.pickle", "rb").read())
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    success_counter = 0

    for i in range(50):
        frame = vs.read()
        frame = imutils.resize(frame, height=480, width=640)

        (h, w) = frame.shape[:2]
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                face = frame[startY:endY, startX:endX]
                face = cv.resize(face, (32, 32))
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]
                if label == 'real': success_counter+=1
                #print(preds)
                #print(j)
                #print(label)

                label = "{}: {:.4f}".format(label, preds[j])
                
                if preds[j] > 0.50 and j == 1:
                    cv.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                    _label = "Liveness: {:.4f}".format(preds[j])
                    cv.putText(frame, _label, (startX, startY - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 0, 255), 2)
                    _label = "Fake: {:.4f}".format(preds[j])
                    cv.putText(frame, _label, (startX, startY - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv.imshow("Frame", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord("q"):
            break
    if success_counter >= 45:
        return(True)
    else:
        return(False)
detector()