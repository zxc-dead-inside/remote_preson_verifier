from deepface import DeepFace
from deepface.commons import distance as dst
import numpy as np


def ver(img1_representation, img2, model_name,
        custom_model, distance_metric='euclidean',
        detector_backend='mediapipe'):
    metric = distance_metric

    img2_representation = DeepFace.represent(
        img_path=img2, model_name=model_name, model=custom_model,
        enforce_detection=True, detector_backend=detector_backend,
        align=False, normalization='base'
        )

    if metric == 'cosine':
        distance = dst.findCosineDistance(
            img1_representation, img2_representation)
    elif metric == 'euclidean':
        distance = dst.findEuclideanDistance(
            img1_representation, img2_representation)
    elif metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(
            img1_representation), dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)

    distance = np.float64(distance)

    if model_name != 'Ensemble':

        threshold = dst.findThreshold(model_name, metric)

        if distance <= threshold:
            identified = True
        else:
            identified = False

        resp_obj = {
            "verified": identified,
            "distance": distance,
            "threshold": threshold,
            "model": model_name,
            "detector_backend": detector_backend,
            "similarity_metric": distance_metric
        }

        return resp_obj
