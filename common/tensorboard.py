from tensorflow.core.framework import summary_pb2

def make_summary_from_python_var(name, val):
    """
    Creates a_t summary from a_t Python variable
    :param name:    Name to be displayed in TensorBoard
    :param val:     Python value
    :return:        Summary Tensor
    """
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name, simple_value=val)])
