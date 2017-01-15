from tflearn.layers.recurrent import BasicLSTMCell, _rnn_template
import tensorflow as tf

from tensorflow.python.ops.rnn_cell import RNNCell, _get_concat_variable, LSTMStateTuple
from tensorflow.python.ops.rnn import rnn, dynamic_rnn
from tensorflow.python.ops.rnn_cell import BasicLSTMCell as tfLSTMCell
import scipy.ndimage as spimage
import numpy as np

from tensorflow.python.platform import tf_logging as logging
import tflearn


def policy_quantization(incoming, n_prototypes, advantage):
    n_features = incoming.get_shape().as_list()[1]

    U_matrix = tf.Variable(tf.eye(n_features), name="U")
    prototypes = tf.Variable(tflearn.initializations.truncated_normal([n_features, n_prototypes]))

    # x is of shape [n_features, batch_size, 1]
    x = tf.expand_dims(tf.matmul(U_matrix, tf.transpose(incoming)), 1)
    # w is of shape [n_features, 1, n_prototypes]
    w = tf.expand_dims(tf.matmul(U_matrix, prototypes), 2)

    # Compute the distance. Note that x - w is of shape [n_features, batch_size, n_prototypes]
    distance = tf.transpose(tf.reduce_sum(tf.square(x - w), axis=0))

    # Get the winning prototypes
    winning_prototypes = tf.argmin(distance, axis=1)

    # Get prototype mask
    prototype_mask = tf.one_hot(winning_prototypes, 1., 0.)
    masked_distance = tf.reduce_sum(distance * prototype_mask, axis=1)

    # Get the sign of the advantage to determine whether this prototype 'matches' the training example
    policy_quantization_loss = tf.reduce_sum(tf.sign(advantage) * masked_distance)

    return winning_prototypes, policy_quantization_loss


class ConvRNNCell(RNNCell):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, outer_filter_size, num_features, strides, inner_filter_size=None, forget_bias=1.0,
                 input_size=None, state_is_tuple=False, activation=tf.nn.tanh, padding='VALID'):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.inner_filter_size = inner_filter_size
        self.outer_filter_size = outer_filter_size
        self.num_features = num_features
        self.strides = strides
        self.padding = padding
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation
        self._num_units = np.prod(shape) * num_features

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicConvLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.

            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(3, 2, state)

            if not self.inner_filter_size:
                concat = _conv_linear([inputs, h], self.outer_filter_size, self.num_features * 4, True)

            else:
                n_input_features = inputs.get_shape().as_list()[-1]

                def get_conv_W(filter_size, n_in_features, n_out_features):
                    return tf.Variable(tflearn.initializations.truncated_normal(
                        [filter_size, filter_size, n_in_features, n_out_features]))
                W_h_to_ijfo = get_conv_W(self.inner_filter_size, self.num_features, self.num_features * 4)
                W_x_to_ijfo = get_conv_W(self.outer_filter_size, n_input_features, self.num_features * 4)
                W_all = tf.concat(0, [tf.reshape(W_h_to_ijfo, [-1]), tf.reshape(W_x_to_ijfo, [-1])], name='Matrix')

                b = tf.Variable(np.concatenate([np.zeros(2 * self.num_features), np.ones(self.num_features),
                                                np.zeros(self.num_features)]), dtype=tf.float32, name='Bias')

                conv_x = tf.nn.conv2d(inputs, W_x_to_ijfo, strides=[1, self.strides, self.strides, 1],
                                      padding=self.padding)
                conv_h = tf.nn.conv2d(h, W_h_to_ijfo, strides=4 * [1], padding='SAME')
                concat = tf.nn.bias_add(conv_x + conv_h, b)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(3, 4, concat)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(3, [new_c, new_h])
            return new_h, new_state


def convolutional_lstm(incoming, outer_filter_size, num_features, stride, inner_filter_size=None, forget_bias=1.0,
                       activation=tf.nn.tanh, padding='VALID', inner_depthwise=False):
    n_input_features = incoming.get_shape().as_list()[-1]

    with tf.name_scope("ConvLSTM"):

        with tf.name_scope("ConvLSTMWeights"):
            def get_conv_W(filter_size, n_in_features, n_out_features, name):
                return tf.Variable(tflearn.initializations.truncated_normal(
                    [filter_size, filter_size, n_in_features, n_out_features]), name=name)

            W_h_to_ijfo = get_conv_W(inner_filter_size, num_features, num_features * 4, name='Wh')
            if inner_depthwise:
                W_x_to_ijfo = get_conv_W(outer_filter_size, n_input_features, 4, name='Wx')
            else:
                W_x_to_ijfo = get_conv_W(outer_filter_size, n_input_features, num_features * 4, name='Wx')

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

                new_c = (c * tf.nn.sigmoid(f + forget_bias) + tf.nn.sigmoid(i) *
                         activation(j))
                new_h = activation(new_c) * tf.nn.sigmoid(o)

                return tf.concat(3, [new_c, new_h])

        with tf.name_scope("Reshaping"):
            _, _, h, w, _ = incoming.get_shape().as_list()
            n_pad = 2 * (outer_filter_size // 2) if padding == 'SAME' else 0
            state_shape = [1, (h - outer_filter_size + n_pad) // stride + 1, (w - outer_filter_size + n_pad) // stride + 1,
                           2 * num_features]
            output_shape = [-1] + state_shape[1:]
            print(state_shape)

            initial_state = tf.placeholder(tf.float32, state_shape)
            new_state = tf.reshape(tf.scan(lstm_step, incoming, initializer=initial_state), output_shape)

            _, outputs = tf.split(3, 2, new_state)

            outputs.W = [W_h_to_ijfo, W_x_to_ijfo]
            outputs.b = b

        return outputs, new_state[-1:], initial_state


def _conv_linear(args, filter_size, num_features, bias, strides, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=strides, padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(3, args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


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


def custom_lstm(incoming, n_units, activation=tf.nn.tanh, forget_bias=1.0, initial_state=None, scope=None,
                name="LSTM", sequence_length=None):

    #with tf.name_scope(name) as scope:
    with tf.variable_scope(name) as vs:
        cell = tfLSTMCell(n_units, forget_bias=forget_bias, activation=activation)#LSTMCell(n_units, forget_bias=forget_bias, activation=activation, name=name)
        o, state = dynamic_rnn(cell, incoming, initial_state=initial_state, sequence_length=sequence_length, scope=vs)
        #o.scope = vs
        vs.reuse_variables()
        o.W = tf.get_variable('BasicLSTMCell/Linear/Matrix')#cell.W
        o.b = tf.get_variable('BasicLSTMCell/Linear/Bias')#cell.b

    return o, state


def conv_lstm(incoming, n_filters, outer_filter_size, stride, inner_filter_size=None, activation=tf.nn.tanh,
              initial_state=None, scope=None, name="LSTM", sequence_length=None, padding='VALID'):

    #with tf.name_scope(name) as scope:
    with tf.variable_scope(name) as vs:
        _, _, h, w, _ = incoming.get_shape().as_list()
        n_pad = 2 * (outer_filter_size // 2) if padding == 'SAME' else 0
        state_shape = [(h - outer_filter_size + n_pad) / stride + 1,
                       (w - outer_filter_size + n_pad) / stride + 1]

        cell = BasicConvLSTMCell(state_shape, outer_filter_size, n_filters, stride, inner_filter_size=inner_filter_size,
                                 state_is_tuple=True, activation=activation)#tfLSTMCell(n_units, forget_bias=forget_bias, activation=activation)#LSTMCell(n_units, forget_bias=forget_bias, activation=activation, name=name)
        o, state = dynamic_rnn(cell, incoming, initial_state=initial_state, sequence_length=sequence_length, scope=vs,
                               time_major=True)
        #o.scope = vs
        vs.reuse_variables()
        o.W = tf.get_variable('BasicConvLSTMCell/Linear/Matrix')#cell.W
        o.b = tf.get_variable('BasicConvLSTMCell/Linear/Bias')#cell.b

    return o, state


def fully_connected_weight_sharing(incoming, axis_len, filter_size, dimensionality='cubic', activation_fn=tf.nn.relu,
                                   weight_decay=0.0, name='FullyConnectedWeightSharing'):

    if dimensionality == 'cubic':
        n_units = axis_len ** 3
        shape = [-1] + 3 * [axis_len] + [1]
    elif dimensionality == 'square':
        n_units = axis_len ** 2
        shape = [-1] + 2 * [axis_len] + [1]
    else:
        raise ValueError("Unknown dimensionality {} for {}, possible options are 'cubic' "
                         "and 'square'.".format(dimensionality, name))

    fc_act = tflearn.layers.fully_connected(incoming, n_units, activation='linear', weight_decay=weight_decay,
                                            bias=False)
    fc_reshaped = tf.reshape(fc_act, shape, name='FC_reshaped')

    if dimensionality == 'square':
        filter = spimage.filters.gaussian_filter(np.ones((filter_size, filter_size)),
                                                 sigma=1.0, mode='constant')
        filter /= np.sum(filter)
        filter_tensor = tf.constant(filter.reshape((filter_size, filter_size, 1, 1)),
                                    dtype=tf.float32, name='FilterTensor')
        fc_locally_weighted = tf.nn.conv2d(fc_reshaped, filter_tensor, [1] * 4, padding='SAME')
    else:
        filter = spimage.filters.gaussian_filter(np.ones((filter_size, filter_size, filter_size)),
                                                 sigma=1.0, mode='constant')
        filter /= np.sum(filter)
        filter_tensor = tf.constant(filter.reshape((filter_size, filter_size, filter_size, 1, 1)),
                                    dtype=tf.float32, name='FilterTensor')
        fc_locally_weighted = tf.nn.conv3d(fc_reshaped, filter_tensor, [1] * 5, padding='SAME')

    bias = tf.Variable(np.zeros(n_units, dtype='float32'))
    o = activation_fn(tf.nn.bias_add(tf.reshape(fc_locally_weighted, [-1, n_units]), bias))
    o.W = fc_act.W
    o.b = bias

    return o


def spatial_weight_sharing(incoming, n_centroids, n_filters, filter_size, strides, activation, name='SoftWeightConv',
                           scope=None, reuse=False, local_normalization=True, centroids_trainable=False, scaling=1.0,
                           distance_fn='EXP', padding='same', sigma_trainable=None):
    """
    Defines a soft weight sharing layer. The soft weight sharing is accomplished by performing multiple convolutions
    which are then combined by a local weighting locally depends on the distance to the 'centroid' of each convolution.
    The centroids are located in between 0 and 1.
    Parameters:
        :param incoming:        Incoming 4D tensor
        :param n_centroids:     Number of centroids (i.e. the number of 'sub'-convolutions to perform)
        :param n_filters:       The number of filters for each convolution
        :param filter_size:     The filter size (assumed to be square)
        :param strides:         The filter stride
        :param activation:      Activation function
        :param name:            The name of this layer
        :param scope:           An optional scope
        :param reuse:           Whether or not to reuse variables here
        :param local_normalization: If True, the distance weighting is locally normalized which serves as a sort of
                    lateral inhibition between competing sub-convolutions. The normalization is divisive.
        :param centroids_trainable: If True, the centroids are trainable. Note that they are concatenated with the
                    conv layers' `b's for compatibility with other tflearn layers
        :param scaling          The scaling factor for the centroids. A scaling factor of 1.0 means that the centroids
                    will be initialized in the range of (0, 1.0)
        :param distance_fn      The distance function that is used to compute the spatial distances of cells w.r.t. the
                                kernel centroids
        :param padding          Padding to use for the sub-convolutions. It is important to have the centroids aligned
                                for stacked spatial weight sharing layers
        :param sigma_trainable  Whether the sigma parameters of the exponential distance function should be trainable.
                    By default, this parameter is set to None, in which case it takes over the value of
                    centroids_trainable.

    Return values:
        :return: A 4D Tensor with similar dimensionality as a normal conv_2d's output
    """

    try:
        vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                   reuse=reuse)
    except Exception:
        vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

    if sigma_trainable is None:
        sigma_trainable = centroids_trainable

    with vscope as scope:
        name = scope.name
        with tf.name_scope("SubConvolutions"):
            convs = [tflearn.conv_2d(incoming, nb_filter=n_filters, filter_size=filter_size, strides=strides,
                                     padding=padding, weight_decay=0., name='SubConv{}'.format(i), activation='linear')
                     for i in range(n_centroids if isinstance(n_centroids, int) else np.prod(n_centroids))]
            stacked_convs = tf.stack(convs, axis=4, name='StackedConvs')

        _, m, n, k = convs[0].get_shape().as_list()

        with tf.name_scope("DistanceWeighting"):
            # First define the x-coordinates per cell. We exploit the TensorFlow's broadcast mechanisms by using
            # single-sized dimensions
            # TODO maybe the linspace should be between 0 and a with a < 1 to make the trainable centroids stable enough
            y_coordinates = tf.reshape(tf.linspace(0., scaling, m), [1, m, 1, 1, 1])
            x_coordinates = tf.reshape(tf.linspace(0., scaling, n), [1, 1, n, 1, 1])

            # Define the centroids variables
            '''centroids_x = tf.Variable(tf.random_uniform([n_centroids], minval=0, maxval=scaling),
                                      trainable=centroids_trainable)
            centroids_y = tf.Variable(tf.random_uniform([n_centroids], minval=0, maxval=scaling),
                                      trainable=centroids_trainable)
            '''
            if isinstance(n_centroids, list):
                assert len(n_centroids) == 2, "Length of n_centroids list must be 2."
                start_x, end_x = scaling / (1 + n_centroids[0]), scaling - scaling / (1 + n_centroids[0])
                start_y, end_y = scaling / (1 + n_centroids[1]), scaling - scaling / (1 + n_centroids[1])
                centroids_x = tf.Variable(tf.concat(0, n_centroids[1] * [tf.linspace(start_x, end_x, n_centroids[0])]))
                centroids_y = tf.Variable(tf.concat(0, n_centroids[0] * [tf.linspace(start_y, end_y, n_centroids[1])]))
                centroids_y = tf.reshape(tf.transpose(tf.reshape(centroids_y, n_centroids)), [np.prod(n_centroids)])
                n_centroids = np.prod(n_centroids)

            elif isinstance(n_centroids, int):
                centroids_x = tf.Variable(tf.random_uniform([n_centroids], minval=0, maxval=scaling),
                                          trainable=centroids_trainable)
                centroids_y = tf.Variable(tf.random_uniform([n_centroids], minval=0, maxval=scaling),
                                          trainable=centroids_trainable)
            else:
                raise TypeError("n_centroids is neither a list nor an int!")

            #centroids_x = tf.Variable(np.asarray([.5, .5], dtype='float32'), trainable=centroids_trainable)
            #centroids_y = tf.Variable(np.asarray([0, 1], dtype='float32'), trainable=centroids_trainable)

            # Define the distance of each cell w.r.t. the centroids. We can easily accomplish this through broadcasting
            # i.e. x_diff2 will have shape [1, 1, n, 1, c] with n as above and c the number of centroids. Similarly,
            # y_diff2 will have shape [1, m, 1, 1, c]
            x_diff2 = tf.square(tf.reshape(centroids_x, [1, 1, 1, 1, n_centroids]) - x_coordinates,
                                name='xDiffSquared')
            y_diff2 = tf.square(tf.reshape(centroids_y, [1, 1, 1, 1, n_centroids]) - y_coordinates,
                                name='yDiffSquared')

            if distance_fn == 'EXP':
                sigma = tf.Variable(scaling/2 * np.ones(n_centroids, dtype='float'), dtype=tf.float32,
                                    name='Sigma', trainable=sigma_trainable)

            # Again, we use broadcasting. The result is of shape [1, m, n, 1, c]
            euclidian_dist = np.sqrt(2) - tf.sqrt(x_diff2 + y_diff2, name="Euclidean") if distance_fn == 'EUCLIDEAN' \
                else tf.exp(-tf.div((x_diff2 + y_diff2), tf.reshape(sigma ** 2, [1, 1, 1, 1, n_centroids])), 'Exp')

            # Optionally, we will perform local normalization such that the weight coefficients add up to 1 for each
            # spatial cell.
            if local_normalization:
                # Add up the distances locally
                total_distance_per_cell = tf.reduce_sum(euclidian_dist, axis=4, keep_dims=True)
                # Now divide
                euclidian_dist = tf.div(euclidian_dist, total_distance_per_cell, name='NormalizedEuclidean')

        with tf.name_scope("SoftWeightSharing"):
            # Compute the distance-weighted output
            dist_weighted = tf.mul(euclidian_dist, stacked_convs, name='DistanceWeighted')

            # Apply non-linearity
            out = activation(tf.reduce_sum(dist_weighted, axis=4), name='Output')

            # Set the variables
            out.W = tf.concat(3, [conv.W for conv in convs], name='W_concat')
            out.b = tf.concat(0,
                              [conv.b for conv in convs] +
                              [centroids_x, centroids_y] +
                              ([sigma] if distance_fn == 'EXP' and sigma_trainable else []), name='b_concatAndCentroids')

            # Add to collection for tflearn functionality
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, out.W)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, out.b)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS + '/' + name, out)

        with tf.name_scope("Summary"):
            euclidean_downsampled = euclidian_dist[:, ::3, ::3, :, :]
            _, m, n, _, _ = euclidean_downsampled.get_shape().as_list()
            euclidian_dist_reshaped = tf.reshape(euclidean_downsampled, 4 * [1] + [m, n, n_centroids])
            current_filter_shape = convs[0].W.get_shape().as_list()
            weights_stacked = tf.reshape(tf.stack([conv.W for conv in convs], axis=4), current_filter_shape + [1, 1, n_centroids])
            locally_weighted_kernels = tf.reduce_sum(tf.mul(euclidian_dist_reshaped, weights_stacked), axis=6)
            locally_weighted_kernels -= tf.reduce_min(locally_weighted_kernels, axis=[4, 5], keep_dims=True)

            in_out_kernel_spatial = tf.transpose(locally_weighted_kernels, [2, 3, 0, 1, 4, 5])
            in_out_kernel_spatial_flattened = tf.pad(tf.reshape(in_out_kernel_spatial, [current_filter_shape[2],
                                                                                        current_filter_shape[3],
                                                                                        current_filter_shape[0],
                                                                                        current_filter_shape[1],
                                                                                        m * n], name='Flattening'),
                                                     [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]])
            current_filter_shape[0] += 1
            current_filter_shape[1] += 1
            in_out_spatial_f_kernel = tf.transpose(in_out_kernel_spatial_flattened, [0, 1, 4, 2, 3])
            in_out_y_reshaped = tf.reshape(in_out_spatial_f_kernel, [current_filter_shape[2],
                                                                     current_filter_shape[3],
                                                                     n,
                                                                     m * current_filter_shape[0],
                                                                     current_filter_shape[1]])
            in_out_y_switched = tf.transpose(in_out_y_reshaped, [0, 1, 2, 4, 3])
            in_out_grid = tf.reshape(in_out_y_switched, [current_filter_shape[2],
                                                         current_filter_shape[3],
                                                         n * current_filter_shape[1],
                                                         m * current_filter_shape[0]])
            summary_image = tf.transpose(in_out_grid, [1, 3, 2, 0])

            out.visual_summary = summary_image
            out.W_list = [conv.W for conv in convs]

    # Add to collection for tflearn functionality
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out)
    return out


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

