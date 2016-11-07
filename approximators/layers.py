from tflearn.layers.recurrent import BasicLSTMCell, _rnn_template
import tensorflow as tf


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

