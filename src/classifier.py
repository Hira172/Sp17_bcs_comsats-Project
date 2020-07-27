import numpy as np
import cv2
import logging

class Classifier:
    def __init__(self, model, logger):
        self.model = model
        self.scale_factor = 0.00392
        self.size = (416, 416)
        self.scale = (0, 0, 0)
        self.swap_rb = True
        self.crop = False
        self.logger = logger

        self.classes = model.get_classes()
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        self.dists = []
        self.TCC = []

        self.xr0 = 0
        self.xl0 = 0
        self.z0 = 1
        self.f = 1.9685  # noraml focal length (inches)

    def get_blob_from_image(self, frame):
        return cv2.dnn.blobFromImage(frame, self.scale_factor,
                                     self.size, self.scale,
                                     self.swap_rb, crop=self.crop)

    def get_box_indices(self):
        return cv2.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)

    def get_distance(self, xl, xr, w):
        global xr0, xl0, z0
        if self.xr0 == 0 and self.xl0 == 0:
            self.xr0 = xr
            self.xl0 = xl
            self.z0 = 1
            return 0
        if self.z0 == 1:
            self.z0 = w * self.f / (self.xr0 - self.xl0)
            return 0
        return (((self.xr0 - self.xl0) * self.z0) / (xr - xl))

    def get_frame_classification(self, fr, height, width, prev_dists):
        class_ids = []
        confidences = []
        boxes = []
        dists = []
        TCC = []

        blob = self.get_blob_from_image(fr)
        self.model.set_input_blob(blob)
        outs = self.model.get_output()

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    # computing distance
                    dist = self.get_distance(x, x+w, w)
                    dists.append(dist)

                    if len(dists) == len(prev_dists):
                        for z1,zo in zip(dists, prev_dists) :
                            if z1 == 0:
                                TCC.append(1000)
                                continue
                            S = self.z0/z1
                            if S == 1:
                                TCC.append(1000)
                                continue
                            Tm = 00.4/ (S-1)    # 0.4 comes from considering 24 frames per sec
                            print(Tm)
                            C = Tm + 1
                            T = Tm *((1-(( 1+2*C)**(1/2))) / C)
                            TCC.append(T)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    self.logger.log(logging.DEBUG, 'class_id: {0}'.format(class_id))

        indexes = self.get_box_indices()
        self.class_ids = class_ids
        self.confidences = confidences
        self.boxes = boxes
        self.dists = dists
        self.TCC = TCC
        return indexes

