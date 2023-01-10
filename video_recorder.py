import cv2 as cv
import mediapipe as mp
from rich.console import Console
import modified_verifier
import liveness_detection


def record(img1_representation, custom_model, model_name, liveness_model, le):
    console = Console()
    right = None
    left = None
    frame_counter = 0
    border_index = 0

    mp_face_detection = mp.solutions.face_detection
    cap = cv.VideoCapture(0)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        positive_recognition = 0
        negative_recognition = 0
        positive_liveness = 0
        negative_liveness = 0
        while True:
            success, frame = cap.read()
            image = frame.copy()
            dst = image.copy()
            dst = cv.flip(dst, 1)
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            h, w, _ = image.shape
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    location = detection.location_data
                    face_cords = location.relative_bounding_box
                    x1 = int(face_cords.xmin*w)
                    y1 = int(face_cords.ymin*h)
                    x2 = int(x1 + face_cords.width*w)
                    y2 = int(y1 + face_cords.height*h)

                    if x1 > 180 and x2 < 460 and y1 > 80 and y2 < 360:
                        border_index = 2
                        if frame_counter <= 100:
                            cv.rectangle(dst, (180, 80),
                                         (460, 360), (0, 255, 0), 2)
                            cv.putText(dst, 'Hold face in the borders',
                                       (100, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            if frame_counter % 2 == 0:
                                face = frame[y1:y2, x1:x2]
                                res = liveness_detection.detector(
                                    face, liveness_model, le)
                                if res:
                                    console.log(
                                        "[green]liveness : real[/green]")
                                    positive_liveness += 1
                                else:
                                    console.log("[red]liveness : fake[/red]")
                                    negative_liveness += 1
                            if frame_counter % 10 == 0:
                                result = modified_verifier.ver(
                                    img1_representation, frame,
                                    model_name, custom_model)
                                if result['verified']:
                                    console.log(
                                        "[green]verified : True[/green]")
                                    positive_recognition += 1
                                else:
                                    console.log("[red]verified : False[/red]")
                                    negative_recognition += 1
                            frame_counter += 1
                        if frame_counter > 100 and right != 'right':
                            cv.rectangle(dst, (180, 80),
                                         (460, 360), (0, 255, 0), 2)
                            cv.putText(
                                dst, 'Turn right', (100, 30),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
                                )
                            face = frame[y1:y2, x1:x2]
                            res = liveness_detection.detector(
                                face, liveness_model, le)
                            if res:
                                console.log("[green]liveness : real[/green]")
                                positive_liveness += 1
                            else:
                                console.log("[red]liveness : fake[/red]")
                                negative_liveness += 1
                            right = head_turn(detection, w)
                        if frame_counter > 100 and left != 'left' and right == 'right':
                            cv.rectangle(dst, (180, 80),
                                         (460, 360), (0, 255, 0), 2)
                            cv.putText(
                                dst, 'Now turn left', (100, 30),
                                cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2
                                )
                            face = frame[y1:y2, x1:x2]
                            res = liveness_detection.detector(
                                face, liveness_model, le)
                            if res:
                                console.log("[green]liveness : real[/green]")
                                positive_liveness += 1
                            else:
                                console.log("[red]liveness : fake[/red]")
                                negative_liveness += 1
                            left = head_turn(detection, w)
                    else:
                        border_index -= 1
                        cv.rectangle(dst, (180, 80),
                                     (460, 360), (0, 0, 255), 2)
                        cv.putText(dst, 'Put your face in the borders',
                                   (100, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv.imshow('VERIFICATION', dst)
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if frame_counter >= 100 and right == "right" and left == "left":
                    cv.destroyAllWindows()

                    if positive_recognition >= 9 and negative_liveness == 0:
                        return True
                    elif positive_recognition >= 9 and negative_liveness/(positive_liveness+negative_liveness) > 0.9:
                        return True
                    else:
                        return False
    cv.destroyAllWindows()


def head_turn(detection, w):
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
    