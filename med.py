import cv2 as cv
import mediapipe as mp

def face_mediapipe(image):
  mp_face_detection = mp.solutions.face_detection
  with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    image_copy = image.copy()
    image_copy.flags.writeable = False
    image_copy = cv.cvtColor(image_copy, cv.COLOR_BGR2RGB)
    h,w,_ = image_copy.shape
    results = face_detection.process(image_copy)
    if results.detections:
      for detection in results.detections:
        location = detection.location_data
        face_cords = location.relative_bounding_box
        x1 = face_cords.xmin*w
        y1 = face_cords.ymin*h
        x2 = x1 + face_cords.width*w
        y2 = y1 + face_cords.height*h
    face = image[int(y1)-w:int(y2)+w,int(x1)-w:int(x2)+w]
    return(face)





