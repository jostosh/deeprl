import logging

logger = logging.getLogger('test')
logger.propagate = False
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(levelname)s] (%(threadName)-10s) - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)