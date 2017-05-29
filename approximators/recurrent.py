import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
try:
    from tensorflow.python.ops.rnn_cell import RNNCell, LSTMStateTuple, BasicLSTMCell
except:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import RNNCell, LSTMStateTuple, BasicLSTMCell
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def ln(tensor, scope = None, epsilon = 1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert(len(tensor.get_shape()) == 4)
    m, v = tf.nn.moments(tensor, [1, 2, 3], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=tensor.get_shape()[1:],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=tensor.get_shape()[1:],
                                initializer=tf.constant_initializer(0))
    LN_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return LN_initial * scale + shift, scale, shift


class ConvLSTM:

    def __init__(self, incoming, outer_filter_size, num_features, stride, inner_filter_size = None, forget_bias = 1.0,
                 activation = tf.nn.tanh, padding = 'VALID', inner_depthwise = False, outer_init = 'torch',
                 inner_init = 'orthogonal', inner_act = tf.nn.sigmoid, name = 'ConvLSTM'):
        n_input_features = incoming.get_shape().as_list()[-1]
        self.vars = []

        with tf.name_scope("ConvLSTM"):

            with tf.name_scope("ConvLSTMWeights"):
                def get_conv_W(filter_size, n_in_features, n_out_features, name, depthwise=False, init='torch'):
                    if init == 'torch':
                        d = 1.0 / np.sqrt(filter_size * filter_size * (n_in_features if not depthwise else 1))
                        weights_init = tf.random_uniform([filter_size, filter_size, n_in_features, n_out_features],
                                                         minval=-d, maxval=d)
                    elif init == 'orthogonal':
                        weights_all = []
                        for i in range(4 if depthwise else 1):
                            X = np.random.random((n_out_features // (4 if depthwise else 1),
                                                  n_in_features * filter_size * filter_size))
                            _, _, Vt = np.linalg.svd(X, full_matrices=False)
                            assert np.allclose(np.dot(Vt, Vt.T), np.eye(Vt.shape[0]))
                            weights_all.append(np.transpose(
                                np.reshape(Vt, (n_out_features, n_in_features, filter_size, filter_size)),
                                (2, 3, 1, 0)
                            ))
                        weights_init = np.concatenate(weights_all, axis=3).astype('float32')
                    else:
                        raise ValueError("Unknown initialization: {}".format(init))
                    return tf.Variable(weights_init, name=name)

                W_x_to_ijfo = get_conv_W(outer_filter_size, n_input_features, num_features * 4, name='Wh', init='torch')
                if inner_depthwise:
                    W_h_to_ijfo = get_conv_W(inner_filter_size, num_features, 4, name='Wx', depthwise=True,
                                             init=inner_init)
                else:
                    W_h_to_ijfo = get_conv_W(inner_filter_size, num_features, num_features * 4, name='Wx',
                                             init=inner_init)

                b = tf.Variable(tf.zeros(4 * num_features), dtype=tf.float32, name='Bias')

            def lstm_step(lstm_state, x):
                with tf.name_scope("ConvLSTMStep"):
                    c, h = tf.split(3, 2, lstm_state)

                    conv_x = tf.nn.conv2d(x, W_x_to_ijfo, strides=[1, stride, stride, 1], padding=padding)
                    if inner_depthwise:
                        conv_h = tf.nn.depthwise_conv2d(h, W_h_to_ijfo, strides=4 * [1], padding='SAME')
                    else:
                        conv_h = tf.nn.conv2d(h, W_h_to_ijfo, strides=4 * [1], padding='SAME')
                    concat = tf.nn.bias_add(conv_x + conv_h, b)

                    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                    i, j, f, o = tf.split(3, 4, concat)

                    i, iscale, ishift = ln(i, scope=name + 'i')
                    j, jscale, jshift = ln(j, scope=name + 'j')
                    f, fscale, fshift = ln(f, scope=name + 'f')
                    o, oscale, oshift = ln(o, scope=name + 'o')
                    self.vars += [iscale, ishift, jscale, jshift, fscale, fshift, oscale, oshift]

                    new_c = (c * inner_act(f + forget_bias) + inner_act(i) * activation(j))
                    new_h = activation(new_c) * inner_act(o)

                    return tf.concat(3, [new_c, new_h])

            with tf.name_scope("Reshaping"):
                _, batch_size, h, w, _ = incoming.get_shape().as_list()
                n_pad = 2 * (outer_filter_size // 2) if padding == 'SAME' else 0
                state_shape = [batch_size,
                               (h - outer_filter_size + n_pad) // stride + 1,
                               (w - outer_filter_size + n_pad) // stride + 1,
                               2 * num_features]
                output_shape = [-1] + state_shape[1:]
                # print(state_shape)

                initial_state = tf.placeholder(tf.float32, state_shape, name='ConvLSTMState')
                new_state = tf.reshape(tf.scan(lstm_step, incoming, initializer=initial_state), output_shape)

                _, outputs = tf.split(3, 2, new_state)

                outputs.W = [W_h_to_ijfo, W_x_to_ijfo]
                outputs.b = b
        self.outputs = outputs
        self.new_state = new_state
        self.initial_state = initial_state
        self.vars += outputs.W + [outputs.b]

    def get_outputs(self):
        return self.outputs, self.new_state[-1:], self.initial_state

    def get_vars(self):
        return self.vars

def convolutional_lstm(incoming, outer_filter_size, num_features, stride, inner_filter_size=None, forget_bias=1.0,
                       activation=tf.nn.tanh, padding='VALID', inner_depthwise=False, outer_init='torch',
                       inner_init='orthogonal', inner_act=tf.nn.sigmoid, name='ConvLSTM'):
    n_input_features = incoming.get_shape().as_list()[-1]

    with tf.name_scope("ConvLSTM"):

        with tf.name_scope("ConvLSTMWeights"):
            def get_conv_W(filter_size, n_in_features, n_out_features, name, depthwise=False, init='torch'):
                if init == 'torch':
                    d = 1.0 / np.sqrt(filter_size * filter_size * (n_in_features if not depthwise else 1))
                    weights_init = tf.random_uniform([filter_size, filter_size, n_in_features, n_out_features],
                                                     minval=-d, maxval=d)
                elif init == 'orthogonal':
                    weights_all = []
                    for i in range(4 if depthwise else 1):
                        X = np.random.random((n_out_features // (4 if depthwise else 1),
                                              n_in_features * filter_size * filter_size))
                        _, _, Vt = np.linalg.svd(X, full_matrices=False)
                        assert np.allclose(np.dot(Vt, Vt.T), np.eye(Vt.shape[0]))
                        weights_all.append(np.transpose(
                            np.reshape(Vt, (n_out_features, n_in_features, filter_size, filter_size)),
                            (2, 3, 1, 0)
                        ))
                    weights_init = np.concatenate(weights_all, axis=3).astype('float32')
                else:
                    raise ValueError("Unknown initialization: {}".format(init))
                return tf.Variable(weights_init, name=name)

            W_x_to_ijfo = get_conv_W(outer_filter_size, n_input_features, num_features * 4, name='Wh', init='torch')
            if inner_depthwise:
                W_h_to_ijfo = get_conv_W(inner_filter_size, num_features, 4, name='Wx', depthwise=True,
                                         init=inner_init)
            else:
                W_h_to_ijfo = get_conv_W(inner_filter_size, num_features, num_features * 4, name='Wx',
                                         init=inner_init)

            b = tf.Variable(tf.zeros(4 * num_features), dtype=tf.float32, name='Bias')

        def lstm_step(lstm_state, x):
            with tf.name_scope("ConvLSTMStep"):
                c, h = tf.split(3, 2, lstm_state)

                conv_x = tf.nn.conv2d(x, W_x_to_ijfo, strides=[1, stride, stride, 1], padding=padding)
                if inner_depthwise:
                    conv_h = tf.nn.depthwise_conv2d(h, W_h_to_ijfo, strides=4 * [1], padding='SAME')
                else:
                    conv_h = tf.nn.conv2d(h, W_h_to_ijfo, strides=4 * [1], padding='SAME')
                concat = tf.nn.bias_add(conv_x + conv_h, b)

                # i = input_gate, j = new_input, f = forget_gate, o = output_gate
                i, j, f, o = tf.split(3, 4, concat)

                i, iscale, ishift = ln(i, scope=name + 'i')
                j, jscale, jshift = ln(j, scope=name + 'j')
                f, fscale, fshift = ln(f, scope=name + 'f')
                o, oscale, oshift = ln(o, scope=name + 'o')

                new_c = (c * inner_act(f + forget_bias) + inner_act(i) * activation(j))
                new_h = activation(new_c) * inner_act(o)

                return tf.concat(3, [new_c, new_h])

        with tf.name_scope("Reshaping"):
            _, batch_size, h, w, _ = incoming.get_shape().as_list()
            n_pad = 2 * (outer_filter_size // 2) if padding == 'SAME' else 0
            state_shape = [batch_size,
                           (h - outer_filter_size + n_pad) // stride + 1,
                           (w - outer_filter_size + n_pad) // stride + 1,
                           2 * num_features]
            output_shape = [-1] + state_shape[1:]
            #print(state_shape)

            initial_state = tf.placeholder(tf.float32, state_shape, name='ConvLSTMState')
            new_state = tf.reshape(tf.scan(lstm_step, incoming, initializer=initial_state), output_shape)

            _, outputs = tf.split(3, 2, new_state)

            outputs.W = [W_h_to_ijfo, W_x_to_ijfo]
            outputs.b = b
            ln_vars = [tf.get_variable]
            outputs.ln = [iscale, ishift, jscale, jshift]

    return outputs, new_state[-1:], initial_state


def convolutional_gru(incoming, outer_filter_size, num_features, stride, inner_filter_size=None, forget_bias=1.0,
                       activation=tf.nn.tanh, padding='VALID', outer_init='torch',
                       inner_init='orthogonal', inner_act=tf.nn.sigmoid):
    n_input_features = incoming.get_shape().as_list()[-1]

    with tf.name_scope("ConvGRU"):

        with tf.name_scope("ConvGRUWeights"):
            def get_conv_W(filter_size, n_in_features, n_out_features, name, depthwise=False, init='torch'):
                if init == 'torch':
                    d = 1.0 / np.sqrt(filter_size * filter_size * (n_in_features if not depthwise else 1))
                    weights_init = tf.random_uniform([filter_size, filter_size, n_in_features, n_out_features],
                                                     minval=-d, maxval=d)
                elif init == 'orthogonal':
                    weights_all = []
                    for i in range(4 if depthwise else 1):
                        X = np.random.random((n_out_features // (3 if depthwise else 1),
                                              n_in_features * filter_size * filter_size))
                        _, _, Vt = np.linalg.svd(X, full_matrices=False)
                        assert np.allclose(np.dot(Vt, Vt.T), np.eye(Vt.shape[0]))
                        weights_all.append(np.transpose(
                            np.reshape(Vt, (n_out_features, n_in_features, filter_size, filter_size)),
                            (2, 3, 1, 0)
                        ))
                    weights_init = np.concatenate(weights_all, axis=3).astype('float32')
                else:
                    raise ValueError("Unknown initialization: {}".format(init))
                return tf.Variable(weights_init, name=name)

            W_x_to_zrh = get_conv_W(outer_filter_size, n_input_features, num_features * 3, name='Wh', init='torch')
            W_h_to_zrh = get_conv_W(inner_filter_size, num_features, num_features * 3, name='Wx',
                                     init=inner_init)

            b = tf.Variable(tf.zeros(3 * num_features), dtype=tf.float32, name='Bias')
            W_h_to_z, W_h_to_r, W_h_to_h = tf.split(3, 3, W_h_to_zrh)

        def gru_step(gru_state_prev, x):
            with tf.name_scope("ConvGRUStep"):
                #c, h = tf.split(3, 2, gru_state)

                # Compute input to hidden convolutions
                conv_x = tf.nn.conv2d(x, W_x_to_zrh, strides=[1, stride, stride, 1], padding=padding)
                conv_x_z, conv_x_r, conv_x_h = tf.split(3, 3, conv_x)

                # Compute hidden to hidden convolutions
                bias_z, bias_r, bias_h = tf.split(0, 3, b)
                conv_s_z = tf.nn.conv2d(gru_state_prev, W_h_to_z, strides=4 * [1], padding='SAME')
                conv_s_r = tf.nn.conv2d(gru_state_prev, W_h_to_r, strides=4 * [1], padding='SAME')
                r = inner_act(tf.nn.bias_add(conv_x_r + conv_s_r, bias_r))
                conv_s_h = tf.nn.conv2d(gru_state_prev * r, W_h_to_h, strides=4*[1], padding='SAME')

                z = inner_act(tf.nn.bias_add(conv_x_z + conv_s_z, bias_z))
                h = activation(tf.nn.bias_add(conv_x_h + conv_s_h * r, bias_h))

                new_gru_state = (1 - z) * h + z * gru_state_prev

                return new_gru_state

        with tf.name_scope("Reshaping"):
            _, batch_size, h, w, _ = incoming.get_shape().as_list()
            n_pad = 2 * (outer_filter_size // 2) if padding == 'SAME' else 0
            state_shape = [batch_size,
                           (h - outer_filter_size + n_pad) // stride + 1,
                           (w - outer_filter_size + n_pad) // stride + 1,
                           num_features]
            output_shape = [-1] + state_shape[1:]
            #print(state_shape)

            initial_state = tf.placeholder(tf.float32, state_shape, name='ConvGRUState')
            outputs = tf.reshape(tf.scan(gru_step, incoming, initializer=initial_state), output_shape)
            outputs.W = [W_h_to_zrh, W_x_to_zrh]
            outputs.b = b

    return outputs, outputs[-1:], initial_state


def custom_lstm(incoming, n_units, activation=tf.nn.tanh, forget_bias=1.0, initial_state=None, scope=None,
                name="LSTM", sequence_length=None):

    #with tf.name_scope(name) as scope:
    with tf.variable_scope(name) as vs:
        cell = BasicLSTMCell(n_units, forget_bias=forget_bias, activation=activation)#LSTMCell(n_units, forget_bias=forget_bias, activation=activation, name=name)
        o, state = dynamic_rnn(cell, incoming, initial_state=initial_state, sequence_length=sequence_length, scope=vs)
        #o.scope = vs
        vs.reuse_variables()
        o.W = tf.get_variable('BasicLSTMCell/Linear/Matrix')#cell.W
        o.b = tf.get_variable('BasicLSTMCell/Linear/Bias')#cell.b

    return o, state

'''
class BasicLSTMCell(RNNCell):
  """Basic LSTM recurrent network cell.

  The implementation is based on: http://arxiv.org/abs/1409.2329.

  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.

  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.

  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tf.nn.tanh):
    """Initialize the basic LSTM cell.

    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      logging.warn("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
      logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = tf.split(1, 2, state)
      concat = _linear([inputs, h], 4 * self._num_units, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(1, 4, concat)

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
               self._activation(j))
      new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat(1, [new_c, new_h])
      return new_h, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: VariableScope for the created subgraph; defaults to "Linear".

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape().as_list() for a in args]
  for shape in shapes:
    if len(shape) != 2:
      raise ValueError("Linear is expecting 2D arguments: %s" % str(shapes))
    if not shape[1]:
      raise ValueError("Linear expects shape[1] of arguments: %s" % str(shapes))
    else:
      total_arg_size += shape[1]

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or "Linear"):
    n_in = args[0].get_shape().as_list()[-1] + args[1].get_shape().as_list()[-1]
    d = 1 / np.sqrt(n_in)
    matrix = tf.get_variable(
        "Matrix", [total_arg_size, output_size], dtype=dtype,
        initializer=tf.random_uniform_initializer(minval=-d, maxval=d)
    )
    if len(args) == 1:
      res = tf.matmul(args[0], matrix)
    else:
      res = tf.matmul(tf.concat(1, args), matrix)
    if not bias:
      return res
    bias_term = tf.get_variable(
        "Bias", [output_size],
        dtype=dtype,
        initializer=tf.random_uniform_initializer(minval=-d, maxval=d))
  return res + bias_term
'''
