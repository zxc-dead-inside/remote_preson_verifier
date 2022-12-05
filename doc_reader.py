import os
import face_cather
import cv2 as cv
from deepface import DeepFace

cur_dir = os.getcwd()

def compare(doc_image):
    doc_image,_,_ = face_cather.cvDnnDetectFaces(image = doc_image, display=False)
    cv.imshow('',doc_image)
    cv.waitKey(0)
    cap = cv.VideoCapture(0)
    success_counter = 0
    for i in range(10):
        ret, frame = cap.read()
        image_height, image_width, _ = frame.shape
        output_image = frame.copy()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        result_im,_,_ = face_cather.cvDnnDetectFaces(output_image, display=False)
        cv.imshow('frame', result_im)
        cv.imshow('doc', doc_image)
        if cv.waitKey(1) == ord('q'):
            break
        result = DeepFace.verify(result_im,doc_image)
        print(result)
        if result['verified'] == True:
            success_counter+=1
    if success_counter>=9:
        return True
    else:
        return False

