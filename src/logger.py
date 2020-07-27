import logging
import sys

class Logger:
    def __init__(self, name=__name__, level=logging.INFO, file='None'):
        self.name = name
        self.level = level
        self.file = file
        self.FORMAT = '%(asctime)-15s %(name)s %(message)s'
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(level)
        self.handler = logging.StreamHandler(sys.stdout)

        if (self.file != 'None'):
            self.file_handler(self.file)

        self.set_format(self.FORMAT)
        self.logger.addHandler(self.handler)

    def file_handler(self, file_name):
        self.handler = logging.FileHandler(file_name)

    def set_format(self, format):
        self.handler.setFormatter(logging.Formatter(format))

    def set_level(self, level):
        self.handler.setLevel(level)


    def log(self, level, msg):
        if level == logging.ERROR:
            self.logger.error(msg)
        elif level == logging.WARN:
            self.logger.warn(msg)
        elif level == logging.DEBUG:
            self.logger.debug(msg)
        else:
            self.logger.info(msg)
