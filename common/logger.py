import logging
from os.path import expanduser
import os

VERSION = 'v0.9.8'
LOGDIRBASE = "/data/s2098407/tensorflowlogs/{}".format(VERSION) #"{}/tensorflowlogs/{}".format(expanduser('~'), VERSION)


def logprefixed(base, hp):
    if hp.logprefix:
        return os.path.join(base, hp.logprefix)
    return base


def get_log_dir(hyper_parameters):

    try:
        path = os.path.join(logprefixed(LOGDIRBASE, hyper_parameters),
                            hyper_parameters.env,
                            hyper_parameters.model,
                            'lpq={}'.format(hyper_parameters.policy_quantization),
                            'glpq={}'.format(hyper_parameters.glvq)
                            )
        # Check if base directory exists, if not create it
        os.makedirs(path, exist_ok=True)
    except PermissionError as e: # "{}/tensorflowlogs/{}".format(expanduser('~'), VERSION)
        basepath = logprefixed(os.path.join(
            os.path.expanduser("~"),
            VERSION
        ), hyper_parameters)
        path = os.path.join(basepath,
                            hyper_parameters.env,
                            hyper_parameters.model,
                            'lpq={}'.format(hyper_parameters.policy_quantization),
                            'glpq={}'.format(hyper_parameters.glvq)
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
