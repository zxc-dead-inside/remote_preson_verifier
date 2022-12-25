import os
import cv2 as cv
from deepface import DeepFace
import med
from rich.console import Console

cur_dir = os.getcwd()

def compare(doc_image):
    console = Console()
    #doc_image,_,_ = face_cather.cvDnnDetectFaces(image = doc_image, display=False)
    #doc_image = med.face_mediapipe(doc_image)
    #cv.imshow('',doc_image)
    #cv.waitKey(0)
    cap = cv.VideoCapture('video.mp4')
    success_counter = 0
    for i in range(10):
        ret, frame = cap.read()
        image_height, image_width, _ = frame.shape
        output_image = frame.copy()
        #result_im,_,_ = face_cather.cvDnnDetectFaces(output_image, display=False)
        #result_im = med.face_mediapipe(output_image)
        result = DeepFace.verify(frame,doc_image)
        if result['verified'] == True:
            success_counter+=1
            console.log("[green]verified : True[/green]")
        else:
            console.log("[red]verified : False[/red]")
    if success_counter>=8:
        return True
    else:
        return False
