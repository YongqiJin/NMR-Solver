from __future__ import absolute_import, division, print_function
import logging
import os
import sys
import datetime
from logging.handlers import TimedRotatingFileHandler
import pytz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class TimeZoneFormatter(logging.Formatter):
    def converter(self, timestamp):
        dt = datetime.datetime.fromtimestamp(timestamp, pytz.utc)
        shanghai_tz = pytz.timezone('Asia/Shanghai')
        shanghai_dt = dt.astimezone(shanghai_tz)
        return shanghai_dt.timetuple()

class PackagePathFilter(logging.Filter):
    def filter(self, record):
        """add relative path to record
        """
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True

class Logger(object):
    def __init__(self, logger_name='None', log_name='unimol_tools'):
        self.logger = logging.getLogger(logger_name)
        logging.root.setLevel(logging.NOTSET)
        self.log_file_name = log_name + '_{0}.log'.format(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))

        cwd_path = os.path.abspath(os.getcwd())
        self.log_path = os.path.join(cwd_path, "logs")

        self.backup_count = 5

        self.console_output_level = 'INFO'
        self.file_output_level = 'INFO'
        self.DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.formatter = TimeZoneFormatter("%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s", self.DATE_FORMAT)

    def get_logger(self) -> logging.Logger:
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
            
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatter)
            console_handler.setLevel(self.console_output_level)
            console_handler.addFilter(PackagePathFilter())
            self.logger.addHandler(console_handler)

            file_handler = TimedRotatingFileHandler(filename=os.path.join(self.log_path, self.log_file_name), when='D',
                        interval=1, backupCount=self.backup_count, delay=True, encoding='utf-8')
            file_handler.setFormatter(self.formatter)
            file_handler.setLevel(self.file_output_level)
            self.logger.addHandler(file_handler)
        return self.logger

    @staticmethod
    def set_file_name(logger_instance, new_log_path, new_log_file_name):
        logger_instance.handlers = [
            h for h in logger_instance.handlers if not isinstance(h, TimedRotatingFileHandler)
        ]
        file_handler = TimedRotatingFileHandler(filename=os.path.join(new_log_path, new_log_file_name), when='D',
                                                interval=1, backupCount=5, delay=True, encoding='utf-8')
        DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
        formatter = TimeZoneFormatter("%(asctime)s | %(relativepath)s | %(lineno)s | %(levelname)s | %(name)s | %(message)s", DATE_FORMAT)
        file_handler.setFormatter(formatter)
        file_handler.setLevel('INFO')
        logger_instance.addHandler(file_handler)

logger = Logger('UniNMR').get_logger()
logger.setLevel(logging.INFO)

