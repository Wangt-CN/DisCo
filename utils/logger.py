# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
from logging import StreamHandler, Handler, getLevelName


# this class is a copy of logging.FileHandler except we end self.close()
# at the end of each emit. While closing file and reopening file after each
# write is not efficient, it allows us to see partial logs when writing to
# fused Azure blobs, which is very convenient
class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        """
        Open the specified file and use it as the stream for logging.
        """
        # Issue #27493: add support for Path objects to be passed in
        filename = os.fspath(filename)
        #keep the absolute path, otherwise derived classes which use this
        #may come a cropper when the current directory changes
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.delay = delay
        if delay:
            #We don't open the stream, but we still need to call the
            #Handler constructor to set level, formatter, lock etc.
            Handler.__init__(self)
            self.stream = None
        else:
            StreamHandler.__init__(self, self._open())

    def close(self):
        """
        Closes the stream.
        """
        self.acquire()
        try:
            try:
                if self.stream:
                    try:
                        self.flush()
                    finally:
                        stream = self.stream
                        self.stream = None
                        if hasattr(stream, "close"):
                            stream.close()
            finally:
                # Issue #19523: call unconditionally to
                # prevent a handler leak when delay is set
                StreamHandler.close(self)
        finally:
            self.release()

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """
        return open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        if self.stream is None:
            self.stream = self._open()
        StreamHandler.emit(self, record)
        self.close()

    def __repr__(self):
        level = getLevelName(self.level)
        return '<%s %s (%s)>' % (
            self.__class__.__name__, self.baseFilename, level)


_LOG_FMT = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s'
_DATE_FMT = '%m/%d/%Y %H:%M:%S'
logging.basicConfig(format=_LOG_FMT, datefmt=_DATE_FMT, level=logging.INFO)
LOGGER = logging.getLogger('__main__')  # this is the global logger
logging.getLogger('matplotlib.font_manager').disabled = True  # cclin
from .common import init_logging
init_logging()

# def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
#     # don't log results for the non-master process
#     if distributed_rank > 0:
#         return logger
#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.INFO)
#     formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     if save_dir:
#         fh = FileHandler(os.path.join(save_dir, filename))
#         fh.setLevel(logging.INFO)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     logging.getLogger('matplotlib.font_manager').disabled = True  # cclin
#     return logger


def add_log_to_file(log_path):
    fh = FileHandler(log_path)
    formatter = logging.Formatter(_LOG_FMT, datefmt=_DATE_FMT)
    fh.setFormatter(formatter)
    LOGGER.addHandler(fh)


from tensorboardX import SummaryWriter

class TensorboardLogger(object):
    def __init__(self):
        self._logger = None
        self._global_step = 0

    def create(self, path):
        self._logger = SummaryWriter(path)

    def noop(self, *args, **kwargs):
        return

    def step(self):
        self._global_step += 1

    @property
    def global_step(self):
        return self._global_step

    @global_step.setter
    def global_step(self, step):
        self._global_step = step

    def log_scalar_dict(self, log_dict, prefix=''):
        """ log a dictionary of scalar values"""
        if self._logger is None:
            return
        if prefix:
            prefix = f'{prefix}_'
        for name, value in log_dict.items():
            if isinstance(value, dict):
                self.log_scalar_dict(value, self._global_step,
                                     prefix=f'{prefix}{name}')
            else:
                self._logger.add_scalar(f'{prefix}{name}', value,
                                        self._global_step)

    def __getattr__(self, name):
        if self._logger is None:
            return self.noop
        return self._logger.__getattribute__(name)


TB_LOGGER = TensorboardLogger()


class RunningMeter(object):
    """ running meteor of a scalar value
        (useful for monitoring training loss)
    """
    def __init__(self, name, val=None, smooth=0.99):
        self._name = name
        self._sm = smooth
        self._val = val

    def __call__(self, value):
        self._val = (value if self._val is None
                     else value*(1-self._sm) + self._val*self._sm)

    def __str__(self):
        return f'{self._name}: {self._val:.4f}'

    @property
    def val(self):
        return self._val

    @property
    def name(self):
        return self._name
