import os
import cv2 as cv
from deepface import DeepFace
import liveness_detection
import doc_reader
import video_recorder
import rich
from rich.console import Console
import argparse

def verification_process(img):
    res = video_recorder.record()
    if res:
        cur_dir = os.getcwd()
        doc_image = cv.imread(cur_dir+'\\'+img)

        res = doc_reader.compare(doc_image)
        cv.destroyAllWindows()

        if res:
            res = liveness_detection.detector()
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str, required=True,
	    help="path to image")
    args = vars(ap.parse_args())
    img = args["image"]
    console = Console()
    if verification_process(img):
        console.log("[green]###############################\n#####VERIFICATION SUCCESSED#####\n###############################[/green]"+"\n"*7)
    else:
        console.log("[red]###############################\n#####VERIFICATION FAILED######\n###############################[/red]"+"\n"*7)