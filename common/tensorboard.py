import tensorflow as tf
import os
from collections import OrderedDict
from deeprl.common.logger import logger, get_log_dir
from tensorflow.core.framework import summary_pb2


def make_summary_from_python_var(name, val):
    """
    Creates a summary from a Python variable
    :param name:    Name to be displayed in TensorBoard
    :param val:     Python value
    :return:        Summary Tensor
    """
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])


def sub_dirs_from_hyper_parameters(hp):
    """
    This function generates the subdir by looking sorting the hyperparameters on keys and joining them with '/'
    :param hp:  The HyperParameters object
    :return:    A subdir to log into
    """
    dict = OrderedDict(sorted(hp.__dict__.items()))
    return '/'.join([(param[:5] if len(param) > 5 else param) + '={}'.format(val) if type(val) is not str else val for param, val in dict.items()])


def writer_new_event(hyper_parameters, session):
    """
    Returns the new event path for TensorBoard
    :param base_path:           The base logging path
    :param hyper_parameters:    A HyperParameters object
    :return:                    Returns a SummaryWriter object
    """
    # Get the logging dir
    fulldir = get_log_dir(hyper_parameters)

    logger.info("Writing TensorBoard logs to {}".format(fulldir))
    return tf.train.SummaryWriter(fulldir, session.graph)

