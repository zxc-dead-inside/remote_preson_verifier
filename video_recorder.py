import cv2 as cv
import mediapipe as mp
import time




def record():

  right = None
  left = None
  frame_counter = 0
  border_index = 0
  

  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils
  cap = cv.VideoCapture(0)
  fourcc = cv.VideoWriter_fourcc(*'MP4V')
  out = cv.VideoWriter('video.mp4', fourcc, 30.0, (640,480))
  


  with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5) as face_detection:
    start_time = time.time()
    while time.time() - start_time < 20:
      success, frame = cap.read()
      image = frame.copy()
      dst = image.copy()
      dst = cv.flip(dst,1)
      if not success:
        print("Ignoring empty camera frame.")
        continue

      
      image.flags.writeable = False
      image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
      h,w,_ = image.shape
      results = face_detection.process(image)

      
      image.flags.writeable = True
      image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
      if results.detections:
        for detection in results.detections:
          mp_drawing.draw_detection(image, detection)
          location = detection.location_data
          face_cords = location.relative_bounding_box
          x1 = face_cords.xmin*w
          y1 = face_cords.ymin*h
          x2 = x1 + face_cords.width*w
          y2 = y1 + face_cords.height*h
          
          if x1 > 180 and x2 < 460 and y1 > 80 and y2 < 360:
            border_index=2
            if frame_counter<100:
              cv.rectangle(dst,(180,80),(460,360),(0,255,0),2)
              cv.putText(dst, 'Hold face in the borders', (100, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
              out.write(frame)
              frame_counter+=1
            if frame_counter>=100 and right != "right":
              cv.rectangle(dst,(180,80),(460,360),(0,255,0),2)
              cv.putText(dst, 'Turn right', (100, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
              right = head_turn(detection,w)
            if frame_counter>=100 and left != "left" and right == "right":
              cv.rectangle(dst,(180,80),(460,360),(0,255,0),2)
              cv.putText(dst, 'Now turn left', (100, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
              left = head_turn(detection,w)
          else:
              border_index-=1
              cv.rectangle(dst,(180,80),(460,360),(0,0,255),2)
              cv.putText(dst, 'Put your face in the borders', (100, 30),
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.imshow('VERIFICATION',dst)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        #if border_index == 1:
        #  return False
        if frame_counter >= 100 and right == "right" and left == "left":
          out.release()
          cv.destroyAllWindows()
          return True
  out.release()
  cv.destroyAllWindows()

def head_turn(detection,w):
    location = detection.location_data
    nose_point = location.relative_keypoints[2]
    face = location.relative_bounding_box
    face_quarter = face.width*w//4

    nose_cord = nose_point.x*w
    if nose_cord < face.xmin*w + face_quarter:
        return "right"
    if nose_cord > face.xmin*w + face.width*w - face_quarter:
        return "left"
    return None
