import logging
from os.path import expanduser
import os

VERSION = 'v0.9.5'
LOGDIRBASE = "/home/jvdw/tensorflowlogs/{}".format(VERSION) #"{}/tensorflowlogs/{}".format(expanduser('~'), VERSION)


def get_log_dir(hyper_parameters):
    try:
        path = os.path.join(LOGDIRBASE,
                            hyper_parameters.env,
                            hyper_parameters.model,
                            'prediction={}'.format(hyper_parameters.frame_prediction),
                            'residuals={}'.format(hyper_parameters.residual_prediction),
                            'feedback={}'.format(hyper_parameters.feedback),
                            'ot={}'.format(hyper_parameters.optimality_tightening)
                            )
        # Check if base directory exists, if not create it
        os.makedirs(path, exist_ok=True)
    except PermissionError as e: # "{}/tensorflowlogs/{}".format(expanduser('~'), VERSION)
        path = os.path.join(os.path.expanduser("~"),
                            "tensorflowlogs",
                            VERSION,
                            hyper_parameters.env,
                            hyper_parameters.model,
                            'prediction={}'.format(hyper_parameters.frame_prediction),
                            'residuals={}'.format(hyper_parameters.residual_prediction),
                            'feedback={}'.format(hyper_parameters.feedback),
                            'ot={}'.format(hyper_parameters.optimality_tightening)
                            )
        os.makedirs(path.replace("/data/s2098407", os.path.expanduser("~")), exist_ok=True)

    # Check the current directories in there
    current_dirs = sorted([o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))])
    # We're assuming at this point that we do not exceed 1M runs per version
    if not current_dirs:
        # If there are no dirs yet, start at 0
        rundir = 'run000000'
    else:
        # Otherwise make a new one by incrementing the count
        lastdir     = current_dirs[-1]
        lastrun     = int(lastdir[3:])
        rundir      = "run%06d" % (lastrun + 1,)
    fulldir = os.path.join(path, rundir)
    return fulldir


logger = logging.getLogger('test')
logger.propagate = False
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] {%(asctime)s} (%(threadName)-10s) - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
