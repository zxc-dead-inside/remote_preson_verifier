import pickle
import argparse
from keras.models import load_model
from deepface import DeepFace
from rich.console import Console
import cv2 as cv
import video_recorder


def verification_process(img):
    model_name = "Facenet512"
    custom_model = DeepFace.build_model(model_name)
    print("["+model_name+" model has been build]")
    doc_image = cv.imread(img)
    img1_representation = DeepFace.represent(
        img_path=doc_image, model_name=model_name, model=custom_model,
        enforce_detection=True, detector_backend='mediapipe',
        align=False, normalization='base'
        )
    liveness_model = load_model('liveness_detector_model')
    le = pickle.loads(open("le.pickle", "rb").read())
    res = video_recorder.record(img1_representation, custom_model,
                                model_name, liveness_model, le)
    return res


if __name__ == "__main__":
    print("starting...")
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
                    help="path to image")
    args = vars(ap.parse_args())
    input_image = args["image"]
    console = Console()
    if verification_process(input_image):
        console.log("[green][VERIFICATION SUCCESSED][/green]"+"\n"*7)
    else:
        console.log("[red][VERIFICATION FAILED][/red]"+"\n"*7)
