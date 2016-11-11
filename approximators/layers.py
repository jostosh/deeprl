from tflearn.layers.recurrent import BasicLSTMCell, _rnn_template
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell, _get_concat_variable, LSTMStateTuple
from tensorflow.python.ops.rnn import rnn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

from tensorflow.python.platform import tf_logging as logging


def spatialsoftmax(incoming, epsilon=0.01):
    # Get the incoming dimensions (should be a 4D tensor)
    _, h, w, c = incoming.get_shape().as_list()

    with tf.name_scope('SpatialSoftmax'):
        # First we create a linspace from -1 + epsilon to 1 - epsilon. Epsilon is needed to ensure that the output is
        # actually in the range (-1, 1) and not greater than 1 or smaller than -1.
        #
        # Note that each '1' in the reshape is to enforce broadcasting along that dimension
        cartesian_y = tf.reshape(tf.linspace(-1 + epsilon, 1 - epsilon, h), (1, h, 1, 1), name="CartesianY")
        cartesian_x = tf.reshape(tf.linspace(-1 + epsilon, 1 - epsilon, w), (1, 1, w, 1), name="CartesianX")

        # Compute the softmax numerator
        numerator_softmax = tf.exp(incoming, name='Numerator')
        # The denominator is computed by computing the sum per channel
        # Again, the '1's in the reshaping are to ensure broadcasting along those dimensions
        denominator_softmax = tf.reshape(tf.reduce_sum(numerator_softmax, reduction_indices=[1, 2]), (-1, 1, 1, c),
                                         name='Denominator')
        # Now we compute the softmax per channel
        softmax_per_channel = tf.div(numerator_softmax, denominator_softmax, name='SoftmaxPerChannel')

        # Compute the x coordinates by element-wise multiplicatoin of the cartesion coordinates with the softmax
        # activations and summing the result
        x_coordinates = tf.reduce_sum(tf.mul(cartesian_x, softmax_per_channel), reduction_indices=[1, 2], name='xOut')
        y_coordinates = tf.reduce_sum(tf.mul(cartesian_y, softmax_per_channel), reduction_indices=[1, 2], name='yOut')

        # Concatenate the resulting tensors to get the output
        o = tf.concat(1, [x_coordinates, y_coordinates], "Output")

    return o


class LSTMCell(RNNCell):
    """Long short-term memory unit (LSTM) recurrent network cell.

    The default non-peephole implementation is based on:

      http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

    S. Hochreiter and J. Schmidhuber.
    "Long Short-Term Memory". Neural Computation, 9(8):1735-1780, 1997.

    The peephole implementation is based on:

      https://research.google.com/pubs/archive/43905.pdf

    Hasim Sak, Andrew Senior, and Francoise Beaufays.
    "Long short-term memory recurrent neural network architectures for
     large scale acoustic modeling." INTERSPEECH, 2014.

    The class uses optional peep-hole connections, optional cell clipping, and
    an optional projection layer.
    """

    def __init__(self, num_units, input_size=None,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=1, num_proj_shards=1,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=tanh, name=None):
        """Initialize the parameters for an LSTM cell.

        Args:
          num_units: int, The number of units in the LSTM cell
          input_size: Deprecated and unused.
          use_peepholes: bool, set True to enable diagonal/peephole connections.
          cell_clip: (optional) A float value, if provided the cell state is clipped
            by this value prior to the cell output activation.
          initializer: (optional) The initializer to use for the weight and
            projection matrices.
          num_proj: (optional) int, The output dimensionality for the projection
            matrices.  If None, no projection is performed.
          proj_clip: (optional) A float value.  If `num_proj > 0` and `proj_clip` is
          provided, then the projected values are clipped elementwise to within
          `[-proj_clip, proj_clip]`.
          num_unit_shards: How to split the weight matrix.  If >1, the weight
            matrix is stored across num_unit_shards.
          num_proj_shards: How to split the projection matrix.  If >1, the
            projection matrix is stored across num_proj_shards.
          forget_bias: Biases of the forget gate are initialized by default to 1
            in order to reduce the scale of forgetting at the beginning of
            the training.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  This latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        if not state_is_tuple:
            logging.warn("%s: Using a concatenated state is slower and will soon be "
                         "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self.name = name

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        """Run one step of LSTM.

        Args:
          inputs: input Tensor, 2D, batch x num_units.
          state: if `state_is_tuple` is False, this must be a state Tensor,
            `2-D, batch x state_size`.  If `state_is_tuple` is True, this must be a
            tuple of state Tensors, both `2-D`, with column sizes `c_state` and
            `m_state`.
          scope: VariableScope for the created subgraph; defaults to "LSTMCell".

        Returns:
          A tuple containing:
          - A `2-D, [batch x output_dim]`, Tensor representing the output of the
            LSTM after reading `inputs` when previous state was `state`.
            Here output_dim is:
               num_proj if num_proj was set,
               num_units otherwise.
          - Tensor(s) representing the new state of LSTM after reading `inputs` when
            the previous state was `state`.  Same type and shape(s) as `state`.

        Raises:
          ValueError: If input size cannot be inferred from inputs via
            static shape inference.
        """
        num_proj = self._num_units if self._num_proj is None else self._num_proj

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]
        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        with vs.variable_scope(scope or type(self).__name__,
                               initializer=self._initializer):  # "LSTMCell"
            concat_w = _get_concat_variable(
                self.name + "_W" if self.name else "W", [input_size.value + num_proj, 4 * self._num_units],
                dtype, self._num_unit_shards)

            b = vs.get_variable(
                self.name + "_B" if self.name else "B", shape=[4 * self._num_units],
                initializer=init_ops.zeros_initializer, dtype=dtype)

            self.W = concat_w
            self.b = b

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            cell_inputs = array_ops.concat(1, [inputs, m_prev])
            lstm_matrix = nn_ops.bias_add(math_ops.matmul(cell_inputs, concat_w), b)
            i, j, f, o = array_ops.split(1, 4, lstm_matrix)

            # Diagonal connections
            if self._use_peepholes:
                w_f_diag = vs.get_variable(
                    "W_F_diag", shape=[self._num_units], dtype=dtype)
                w_i_diag = vs.get_variable(
                    "W_I_diag", shape=[self._num_units], dtype=dtype)
                w_o_diag = vs.get_variable(
                    "W_O_diag", shape=[self._num_units], dtype=dtype)

            if self._use_peepholes:
                c = (sigmoid(f + self._forget_bias + w_f_diag * c_prev) * c_prev +
                     sigmoid(i + w_i_diag * c_prev) * self._activation(j))
            else:
                c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                     self._activation(j))

            if self._cell_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
                # pylint: enable=invalid-unary-operand-type

            if self._use_peepholes:
                m = sigmoid(o + w_o_diag * c) * self._activation(c)
            else:
                m = sigmoid(o) * self._activation(c)

            if self._num_proj is not None:
                concat_w_proj = _get_concat_variable(
                    "W_P", [self._num_units, self._num_proj],
                    dtype, self._num_proj_shards)

                m = math_ops.matmul(m, concat_w_proj)
                if self._proj_clip is not None:
                    # pylint: disable=invalid-unary-operand-type
                    m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                    # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple
                     else array_ops.concat(1, [c, m]))
        return m, new_state


def unpack_sequence(tensor):
    """Split the single tensor of a sequence into a list of frames."""
    return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))


def pack_sequence(sequence):
    """Combine a list of the frames into a single tensor of the sequence."""
    return tf.transpose(tf.pack(sequence), perm=[1, 0, 2])


def custom_lstm(incoming, n_units, activation=tf.nn.tanh, forget_bias=1.0, initial_state=None, scope=None,
                name="LSTM", sequence_length=None):

    with tf.name_scope(name):
        cell = LSTMCell(n_units, forget_bias=forget_bias, activation=activation, name=name)

        o, state = rnn(cell, unpack_sequence(incoming), initial_state=initial_state, sequence_length=sequence_length,
                       scope=scope)
        o = pack_sequence(o)

    o.scope = scope
    o.W = cell.W
    o.b = cell.b

    return o, state


def lstm(incoming, n_units, activation='tanh', inner_activation='sigmoid',
         dropout=None, bias=True, weights_init=None, forget_bias=1.0,
         return_seq=False, return_state=False, initial_state=None,
         dynamic=False, trainable=True, restore=True, reuse=False,
         scope=None, name="LSTM"):
    """ LSTM.

    Long Short Term Memory Recurrent Layer.

    Input:
        3-D Tensor [samples, timesteps, input dim].

    Output:
        if `return_seq`: 3-D Tensor [samples, timesteps, output dim].
        else: 2-D Tensor [samples, output dim].

    Arguments:
        incoming: `Tensor`. Incoming 3-D Tensor.
        n_units: `int`, number of units for this layer.
        activation: `str` (name) or `function` (returning a `Tensor`).
            Activation applied to this layer (see tflearn.activations).
            Default: 'tanh'.
        inner_activation: `str` (name) or `function` (returning a `Tensor`).
            LSTM inner activation. Default: 'sigmoid'.
        dropout: `tuple` of `float`: (input_keep_prob, output_keep_prob). The
            input and output keep probability.
        bias: `bool`. If True, a bias is used.
        weights_init: `str` (name) or `Tensor`. Weights initialization.
            (See tflearn.initializations).
        forget_bias: `float`. Bias of the forget gate. Default: 1.0.
        return_seq: `bool`. If True, returns the full sequence instead of
            last sequence output only.
        return_state: `bool`. If True, returns a tuple with output and
            states: (output, states).
        initial_state: `Tensor`. An initial state for the RNN.  This must be
            a tensor of appropriate type and shape [batch_size x cell.state_size].
        dynamic: `bool`. If True, dynamic computation is performed. It will not
            compute RNN steps above the sequence length. Note that because TF
            requires to feed sequences of same length, 0 is used as a mask.
            So a sequence padded with 0 at the end must be provided. When
            computation is performed, it will stop when it meets a step with
            a value of 0.
        trainable: `bool`. If True, weights will be trainable.
        restore: `bool`. If True, this layer weights will be restored when
            loading a model.
        reuse: `bool`. If True and 'scope' is provided, this layer variables
            will be reused (shared).
        scope: `str`. Define this layer scope (optional). A scope can be
            used to share variables between layers. Note that scope will
            override name.
        name: `str`. A name for this layer (optional).

    References:
        Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber,
        Neural Computation 9(8): 1735-1780, 1997.

    Links:
        [http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf]
        (http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)

    """
    cell = BasicLSTMCell(n_units, activation=activation,
                         inner_activation=inner_activation,
                         forget_bias=forget_bias, bias=bias,
                         weights_init=weights_init, trainable=trainable,
                         restore=restore, reuse=reuse)
    x = _rnn_template(incoming, cell=cell, dropout=dropout,
                      return_seq=return_seq, return_state=return_state,
                      initial_state=initial_state, dynamic=dynamic,
                      scope=scope, name=name)

    o, state = x
    if return_seq:
        o = tf.concat(0, o, name="StackedSequence")

    #print(o)
    o.scope = scope
    o.W = cell.W
    o.b = cell.b
    return o, state

