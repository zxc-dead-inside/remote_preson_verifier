from tensorflow.keras.utils import img_to_array
import numpy as np
import cv2 as cv


def detector(face, liveness_model, le):

    face = cv.resize(face, (32, 32))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = liveness_model.predict(face)[0]

    j = np.argmax(preds)
    label = le.classes_[j]
    if label == 'real':
        return True
    else:
        return False
