import tensorflow as tf
import os
from collections import OrderedDict
from deeprl.common.logger import logger


def get_sub_dirs_from_hyper_parameters(hp):
    """
    This function generates the subdir by looking sorting the hyperparameters on keys and joining them with '/'
    :param hp:  The HyperParameters object
    :return:    A subdir to log into
    """
    dict = OrderedDict(sorted(hp.__dict__.items()))
    return '/'.join([(param[:5] if len(param) > 5 else param) + '={}'.format(val) for param, val in dict.items()])


def get_writer_new_event(base_path, hyper_parameters):
    """
    Returns the new event path for TensorBoard
    :param base_path:           The base logging path
    :param hyper_parameters:    A HyperParameters object
    :return:                    Returns a SummaryWriter object
    """
    # Check if base directory exists, if not create it
    os.makedirs(base_path, exist_ok=True)

    # Check the current directories in there
    current_dirs = sorted([o for o in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, o))])
    # We're assuming at this point that we do not exceed 1M runs per version
    if not current_dirs:
        # If there are no dirs yet, start at 0
        rundir = 'run000000'
    else:
        # Otherwise make a new one by incrementing the count
        lastdir     = current_dirs[-1]
        lastrun     = int(lastdir[3:])
        rundir      = "run%06d" % (lastrun + 1,)
    fulldir = os.path.join(base_path, rundir, get_sub_dirs_from_hyper_parameters(hyper_parameters))

    logger.info("Writing TensorBoard logs to {}".format(fulldir))
    return tf.train.SummaryWriter(fulldir, tf.get_default_graph())

