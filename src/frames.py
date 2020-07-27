import cv2
import numpy as np
import logging

class Frames:
    def __init__(self, mp4_file_name, logger):
        self.file_mp4 = mp4_file_name
        self.logger = logger
        self.frame_capture = ""
        self.colors = []
        self.font = cv2.FONT_HERSHEY_PLAIN

    def open_frames_file(self):
        self.frame_capture = cv2.VideoCapture(self.file_mp4)
        if (self.frame_capture.isOpened() == False):
            self.logger.log(logging.ERROR, 'frame capture failed from file {0}'.format(self.file_mp4))
            return -1
        return 0

    def is_opened(self):
        return self.frame_capture.isOpened()

    def read_frame(self):
        ret, fr = self.frame_capture.read()
        return ret, fr

    def get_frame_shape(self, fr):
        height, width, channels = fr.shape
        return height, width, channels

    def set_class_colors(self, classes):
        self.colors = np.random.uniform(0, 100, size=(len(classes), 3))

    def get_color(self, i):
        return self.colors[i]

    def draw_rect(self, fr, x, y, color, thickness):
        cv2.rectangle(fr, x, y, color, thickness)

    def put_text(self, fr, text, loc, font, font_scale=1, color=(255, 255, 255)):
        cv2.putText(fr, text, loc, font, font_scale, color)

    def display_frame(self, fr):
        cv2.imshow('Frame', fr)

    def display_continous(self):
        if cv2.waitKey(25) & 0xFF == ord('q'):
            return

    def display_wait(self):
        if cv2.waitKey() & 0xFF == ord('q'):
            return

    def release_frames(self):
        self.frame_capture.release()

    def destroy_all_windows(self):
        cv2.destroyAllWindows()

    def finalize(self):
        self.release_frames()
        self.destroy_all_windows()