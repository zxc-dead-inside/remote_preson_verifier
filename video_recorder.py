import cv2 as cv
import imutils
from imutils.video import VideoStream
import numpy as np
import time 
import face_cather


def record():
    check_index = 0 
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    fourcc = cv.VideoWriter_fourcc(*'MP4V')
    out = cv.VideoWriter('video.mp4', fourcc, 30.0, (640,480))
    right = cv.VideoWriter('turn_right.mp4', fourcc, 30.0, (640,480))
    left = cv.VideoWriter('turn_left.mp4', fourcc, 30.0, (640,480))
    start_time = time.time()
    while time.time() - start_time < 5:
        frame = vs.read()
        frame = imutils.resize(frame, height=480, width=640)
        dst = frame.copy()
        _,piska,face_cords = face_cather.cvDnnDetectFaces(frame)
        for face in face_cords[0][0]:
            face_confidence = face[2]
            if face_confidence > 0.5:
                bbox = face[3:]
                x1 = int(bbox[0] * 640)
                y1 = int(bbox[1] * 480)
                x2 = int(bbox[2] * 640)
                y2 = int(bbox[3] * 480)
                x2-x1//2
        if x1 > 180 and x2 < 460 and y1 > 80 and y2 < 360:
            check_index=2
            cv.rectangle(dst,(180,80),(460,360),(0,255,0),2)
            cv.putText(dst, 'Hold face in the borders', (100, 30),
		        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            out.write(frame)
        else:
            check_index-=1
            cv.rectangle(dst,(180,80),(460,360),(0,0,255),2)
            cv.putText(dst, 'Put your face in the borders', (100, 30),
		        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if check_index==1:
            return(False)
        cv.imshow("Frame", dst)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    return(True)

    out.release()
    cv.destroyAllWindows()
    vs.stop()
record()