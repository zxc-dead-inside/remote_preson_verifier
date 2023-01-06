import os
import cv2 as cv
from deepface import DeepFace
import med
from rich.console import Console
import time
import modified_verifier

cur_dir = os.getcwd()

def compare(doc_image):
    console = Console()
    #doc_image,_,_ = face_cather.cvDnnDetectFaces(image = doc_image, display=False)
    #doc_image = med.face_mediapipe(doc_image)
    #cv.imshow('',doc_image)
    #cv.waitKey(0)
    tic = time.time()
    model_name = "Facenet"
    custom_model = DeepFace.build_model(model_name)
    img1_representation = DeepFace.represent(img_path = doc_image
                            , model_name = model_name, model = custom_model
                            , enforce_detection = True, detector_backend = 'mediapipe'
                            , align = False
                            , normalization = 'base'
                            )
    cap = cv.VideoCapture('video.mp4')
    success_counter = 0
    for i in range(10):
        ret, frame = cap.read()
        image_height, image_width, _ = frame.shape
        output_image = frame.copy()
        #result_im,_,_ = face_cather.cvDnnDetectFaces(output_image, display=False)
        #result_im = med.face_mediapipe(output_image)
        t=time.time()
        result = modified_verifier.ver(img1_representation, frame, model_name, custom_model)
        #result = DeepFace.verify(doc_image,frame,model_name='Facenet',distance_metric='euclidean')
        print(t - time.time())
        if result['verified'] == True:
            success_counter+=1
            console.log("[green]verified : True[/green]")
        else:
            console.log("[red]verified : False[/red]")
    toc = time.time()
    print(toc-tic)
    if success_counter>=8:
        return True
    else:
        return False


compare("myself.jpg")