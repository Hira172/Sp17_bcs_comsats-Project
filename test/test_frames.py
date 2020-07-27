import unittest
import logging

from src.frames import Frames
from src.logger import Logger

class MyTestCase(unittest.TestCase):
    def test_frames(self):
        l = Logger('test_frames')
        f = Frames('../Test2.mp4', l)
        height = 0
        width = 0
        if (f.open_frames_file() == 0):
            ret, frame = f.read_frame()
            if ret == True:
                height, width, channels = f.get_frame_shape(frame)
                l.log(logging.INFO, 'frame shape: height: {0}, width: {1}'.format(height, width))
                f.display_frame(frame)
                f.display_wait()
            else:
                l.log(logging.ERROR, 'frame read failed')
        self.assertNotEqual(height, 0)
        self.assertNotEqual(width, 0)


if __name__ == '__main__':
    unittest.main()
