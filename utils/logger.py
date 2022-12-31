import logging
import os
import sys


class Logger:
    logger = None

    @staticmethod
    def get_logger(filename: str = None):
        if not Logger.logger:
            Logger.init_logger(filename=filename)
        return Logger.logger

    @staticmethod
    def init_logger(
            level=logging.INFO,
            fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            filename: str = None):
        logger = logging.getLogger(filename)
        logger.setLevel(level)
        fmt = logging.Formatter(fmt)
        
        if os.path.exists(filename):
            os.remove(filename)

        # file handler
        fh = logging.FileHandler(filename)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # stream handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.setLevel(level)
        Logger.logger = logger
        return logger
