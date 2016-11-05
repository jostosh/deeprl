from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops.gen_array_ops import *


def sequence_mask(lengths, maxlen=None, dtype=dtypes.bool, name=None):
  """Return a mask tensor representing the first N positions of each row.

  Example:
  ```python
  tf.sequence_mask([1, 3, 2], 5) =
    [[True, False, False, False, False],
     [True, True, True, False, False],
     [True, True, False, False, False]]
  ```

  Args:
    lengths: 1D integer tensor, all its values < maxlen.
    maxlen: scalar integer tensor, maximum length of each row. Default: use
            maximum over lengths.
    dtype: output type of the resulting tensor.
    name: name of the op.
  Returns:
    A 2D mask tensor, as shown in the example above, cast to specified dtype.

  Raises:
    ValueError: if the arguments have invalid rank.
  """
  with ops.name_scope(name, "SequenceMask"):
    lengths = ops.convert_to_tensor(lengths)
    if lengths.get_shape().ndims != 1:
      raise ValueError("lengths must be 1D for sequence_mask")

    if maxlen is None:
      maxlen = gen_math_ops._max(lengths, [0])
    else:
      maxlen = ops.convert_to_tensor(maxlen)
    if maxlen.get_shape().ndims != 0:
      raise ValueError("maxlen must be scalar for sequence_mask")

    # The basic idea is to compare a range row vector of size maxlen:
    # [0, 1, 2, 3, 4]
    # to length as a matrix with 1 column: [[1], [3], [2]].
    # Because of broadcasting on both arguments this comparison results
    # in a matrix of size (len(lengths), maxlen)
    result = gen_math_ops._range(0, maxlen, 1) < expand_dims(lengths, 1)
    if dtype is None or result.dtype.base_dtype == dtype.base_dtype:
      return result
    else:
      return gen_math_ops.cast(result, dtype)
