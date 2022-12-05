import os
import face_cather
import cv2 as cv
from deepface import DeepFace
import liveness_detection
import doc_reader

cur_dir = os.getcwd()
doc_image = cv.imread(cur_dir+'\\Tolik.jpg')

res = doc_reader.compare(doc_image)
cv.destroyAllWindows()

if res == True:
    res = liveness_detection.detector()
print(res)