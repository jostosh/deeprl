import logging

logger = logging.getLogger('test')
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] {%(asctime)s} (%(threadName)-10s) - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
