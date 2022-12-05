import numpy as np
from time import time
import cv2 as cv
import matplotlib.pyplot as plt

opencv_dnn_model = cv.dnn.readNetFromTensorflow(model="face_detector_model/opencv_face_detector_uint8.pb", 
    config="face_detector_model/opencv_face_detector.pbtxt")

def cvDnnDetectFaces(image, opencv_dnn_model = opencv_dnn_model, min_confidence=0.5, display = False):
    
    
    image_height, image_width, _ = image.shape
    presentation = image.copy()
    preprocessed_image = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    opencv_dnn_model.setInput(preprocessed_image)
    results = opencv_dnn_model.forward()    
    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)
            width = (x2 - x1) // 5
            heiht = (y2 - y1) // 5
            face_image = presentation[y1-heiht:y2+heiht,x1-width:x2+width].copy()
            cv.rectangle(presentation, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//200)
            cv.rectangle(presentation, pt1=(x1, y1-image_width//20), pt2=(x1+image_width//16, y1),
                          color=(0, 255, 0), thickness=-1)
            cv.putText(presentation, text=str(round(face_confidence, 1)), org=(x1, y1-25), 
                        fontFace=cv.FONT_HERSHEY_COMPLEX, fontScale=image_width//700,
                        color=(255,255,255), thickness=image_width//200)
    if display == True:
        cv.imshow('',presentation)
        cv.waitKey(0)
    return face_image, presentation, results 