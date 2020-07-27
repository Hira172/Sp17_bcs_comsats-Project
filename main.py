from src.logger import Logger
from src.frames import Frames
from src.model import Model
from src.classifier import Classifier

import numpy as np
import logging

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
        for i in range(len(classifier.boxes)):
            if i in indexes:
                x, y, w, h = classifier.boxes[i]
                label = str(classifier.classes[classifier.class_ids[i]]) + " " + str(round(classifier.dists[i],3))
                if i < len(classifier.TCC):
                    label = label+"  " + str(classifier.TCC[i])
                color = model.colors[i]
                f.draw_rect(frame, (x, y), (x + w, y + h), color, 2)
                f.draw_rect(frame, (x, y-10), (x + w, y), color, -1)
                f.put_text(frame, label, (x,y), f.font, 1, (255,255,255))

        prev_dists = classifier.dists

        # display frame with boxes
        f.display_frame(frame)
        f.display_continous()

    f.finalize()

if __name__ == '__main__':
    main()