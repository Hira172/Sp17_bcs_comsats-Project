import unittest
import logging

from src.logger import Logger

class MyTestCase(unittest.TestCase):
    def test_logger_info(self):
        logger = Logger("logger_info_test")
        logger.log(logging.INFO, "this is a test of sev info")
        logger.log(logging.DEBUG, "this is a debug msg test of sev debug")
        self.assertEqual(True, True)

    def test_logger_debug(self):
        logger = Logger("logger_debug_test", logging.DEBUG)
        logger.log(logging.INFO, "this is a test of sev info")
        logger.log(logging.DEBUG, "this is a debug msg test of sev debug")
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
