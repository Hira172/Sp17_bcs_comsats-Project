import unittest
import logging

from src.model import Model
from src.logger import Logger


class MyTestCase(unittest.TestCase):
    def test_model(self):
        l = Logger('test_model')
        m = Model('../yolov3.weights', '../yolov3.cfg', '../coco.names', l)
        m.load_classes_from_file()
        m.setup_dnn()
        self.assertNotEqual(m.get_net(), '')


if __name__ == '__main__':
    unittest.main()
