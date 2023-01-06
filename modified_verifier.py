from deepface import DeepFace
import time
from deepface.commons import functions, realtime, distance as dst
import cv2 as cv
import mediapipe as mp
import numpy as np


def ver(img1_representation,img2,model_name,custom_model,distance_metric = 'euclidean',detector_backend = 'mediapipe'):
    metric = distance_metric
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    enforce_detection = True
    resp_objects = []
    

    #result_im,_,_ = face_cather.cvDnnDetectFaces(output_image, display=False)
    #result_im = med.face_mediapipe(output_image)

    img2_representation = DeepFace.represent(img_path = img2
                        , model_name = model_name, model = custom_model
                        , enforce_detection = True, detector_backend = detector_backend
                        , align = False
                        , normalization = 'base'
                        )


    if metric == 'cosine':
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distance = np.float64(distance) #causes trobule for euclideans in api calls if this is not set (issue #175)
    #----------------------
    #decision

    if model_name != 'Ensemble':

        threshold = dst.findThreshold(model_name, metric)

        if distance <= threshold:
            identified = True
        else:
            identified = False

        resp_obj = {
            "verified": identified
            , "distance": distance
            , "threshold": threshold
            , "model": model_name
            , "detector_backend": detector_backend
            , "similarity_metric": distance_metric
        }

        return(resp_obj)



# cap = cv.VideoCapture('video.mp4')
# model_name = "Facenet"
# custom_model = DeepFace.build_model(model_name)
# img1_representation = DeepFace.represent(img_path = "myself.jpg"
#                             , model_name = model_name, model = custom_model
#                             , enforce_detection = True, detector_backend = 'mediapipe'
#                             , align = True
#                             , normalization = 'base'
#                             )
# tic = time.time()
# for i in range(10):
#     ret, frame = cap.read()
#     image_height, image_width, _ = frame.shape
#     output_image = frame.copy()
#     ver(img1_representation, frame,model_name,custom_model)
# toc = time.time()
# print(toc-tic)

