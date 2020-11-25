from flask import Flask,render_template,jsonify,request, redirect
import cv2
from yolo.utils import Load_Yolo_model
import sys, getopt
from core import core
from yolo.configs import YOLO_INPUT_SIZE


global yolo
yolo = Load_Yolo_model()
core = core()

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")



@app.route("/VideoForm", methods = ['POST'])
def videoForm():
    video_path = request.form['fileInput']
    video_path = "Videos\\" + video_path
    res = core.system(yolo, video_path, "detection.mp4", input_size=YOLO_INPUT_SIZE, show=True, iou_threshold=0.1,
           rectangle_colors=(255, 0, 0), Track_only=['car', 'truck', 'motorbike', 'person'], display_tm=True, realTime=False)
    return render_template("Video.html") if res else redirect('/');


@app.route("/getImg")
def getImg():
    return core.getImg()

if __name__ == "__main__":
    app.run()