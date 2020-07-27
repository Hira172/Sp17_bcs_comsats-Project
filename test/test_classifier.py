import unittest

from src.logger import Logger
from src.frames import Frames
from src.model import Model
from src.classifier import Classifier


class MyTestCase(unittest.TestCase):
    def test_classifier(self):
        l = Logger('test_classifier')
        m = Model('../yolov3.weights', '../yolov3.cfg', '../coco.names', l)
        m.load_classes_from_file()
        m.setup_dnn()
        classifier = Classifier(m, l)

        f = Frames('../Test2.mp4', l)

        prev_dists = ''
        if (f.open_frames_file() == 0):
            ret, frame = f.read_frame()
            if ret == True:
                height, width, channels = f.get_frame_shape(frame)
                indexes = classifier.get_frame_classification(frame, height, width, prev_dists)
        self.assertNotEqual(indexes, '')


if __name__ == '__main__':
    unittest.main()
