import numpy as np
import cv2
import logging

class Model:
    def __init__(self, weights, config, class_file, logger):
        self.weights = weights
        self.config = config
        self.class_file = class_file
        self.logger = logger

        self.net = ''
        self.layer_names = ''
        self.output_layers = ''
        self.colors = ''
        self.classes = ''

    def load_classes_from_file(self):
        with open(self.class_file, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

    def setup_dnn(self):
        self.net = cv2.dnn.readNet(self.weights, self.config)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.colors = np.random.uniform(0, 100, size=(len(self.classes), 3))

    def set_input_blob(self, blob):
        self.net.setInput(blob)

    def get_output(self):
        return self.net.forward(self.output_layers)

    def get_net(self):
        return self.net

    def get_layer_names(self):
        return self.layer_names

    def get_output_layers(self):
        return self.output_layers

    def get_classes(self):
        return self.classes

