from src.logger import Logger
from src.frames import Frames
from src.model import Model
from src.classifier import Classifier

import numpy as np
import cv2
import logging

# WEIGHTS_FILE = 'D:\\fyp\Codes\\try\Sp17_bcs_comsats-Project\yolov3.weights' # For Hira's system
WEIGHTS_FILE = 'yolov3.weights'
CONFIGS_FILE = 'yolov3.cfg'
CLASSES_FILE = 'coco.names'
VIDEO_FILE = 'Test1.mp4'

def main():
    logger = Logger("demo", logging.DEBUG)
    logger.log(logging.DEBUG, "Starting the demo")

    prev_dists = []

    # setup dnn model
    model = Model(WEIGHTS_FILE, CONFIGS_FILE, CLASSES_FILE, logger)
    model.load_classes_from_file()
    model.setup_dnn()

    # setup classifier
    classifier = Classifier(model, logger)

    # initialize frame object
    f = Frames(VIDEO_FILE, logger)
    if (f.open_frames_file() < 0):
        logger.log(logging.ERROR, 'error opening video stream')
        return -1

    f.set_class_colors(model.classes)

    while(f.is_opened()):
        ret, frame = f.read_frame()
        if ret != True:
            break

        height, width, channels = f.get_frame_shape(frame)
        indexes = classifier.get_frame_classification(frame, height, width, prev_dists)

        # label possibly multiple identified objects in a frame
        i = 0
        # for i in range(len(classifier.boxes)):
        if i in indexes:
            x, y, w, h = classifier.boxes[i]
            label = "Class: "+str(classifier.classes[classifier.class_ids[i]])
            label2 = "distance: "+ str(round(classifier.dists[i],3))
            if i < len(classifier.TCC):
                label3 = "TTC: " + str(round(classifier.TCC[i],3))
            color = model.colors[i]
            f.draw_rect(frame, (x, y), (x + w, y+h), color, 2) # bounding box
            h3 = 3*30
            print("w",w,"h",h,"x",x,"y",y)
            f.draw_rect(frame, (x-85, y-h3), (x +85 + 85, y-30), (0,0,255), -1) # arrow rectangle
            mid_w=85/2
            triangle_cnt = np.array([(x+85+85 , y-30),(x-85, y-30) ,(x+int(mid_w), y)]) # arrow traingle
            cv2.drawContours(frame, [triangle_cnt], 0,  (0,0,255), -1)

            f.put_text(frame, label, (x, y-h3+17), f.font, 1, (255,255,255))
            f.put_text(frame, label2, (x, y - h3 + 37), f.font, 1, (255, 255, 255))
            f.put_text(frame, label3, (x, y - h3 + 57), f.font, 1, (255, 255, 255))

        prev_dists = classifier.dists

        # display frame with boxes
        f.display_frame(frame)
        f.display_continous()

    f.finalize()

if __name__ == '__main__':
    main()