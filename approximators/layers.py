#from tflearn.layers.recurrent import BasicLSTMCell, _rnn_template
import numpy as np
# from tensorflow.python.ops.rnn_cell import BasicLSTMCell as tfLSTMCell
import scipy.ndimage as spimage
import tensorflow as tf
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


def conv_layer(incoming, n_filters, filter_size, stride, activation, name, padding='valid', init='torch',
               bias_init=0.01):
    _, kh, kw, input_channels = incoming.get_shape().as_list()
    d = 1.0 / np.sqrt(filter_size * filter_size * input_channels)
    if init == 'torch':
        weight_init = tf.random_uniform([filter_size, filter_size, input_channels, n_filters], minval=-d, maxval=d)
        bias_init   = tf.constant_initializer(bias_init)
    else:
        weight_init = tf.contrib.layers.variance_scaling_initializer()
        bias_init   = tf.constant_initializer(bias_init)

    return tflearn.conv_2d(incoming=incoming, nb_filter=n_filters, filter_size=filter_size, strides=stride,
                           padding=padding, activation=activation, weights_init=weight_init, bias_init=bias_init,
                           weight_decay=0.0, name=name)


def fc_layer(incoming, n_out, activation, name, init='torch', bias_init=0.01):
    _, n_in = incoming.get_shape().as_list()
    d = 1.0 / np.sqrt(n_in)
    if init == 'torch':
        weights_init = tf.random_uniform([n_in, n_out], minval=-d, maxval=d)
        bias_initializer = tf.constant_initializer(bias_init)
    else:
        weights_init = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer    = tf.constant_initializer(bias_init)

    return tflearn.fully_connected(incoming=incoming, n_units=n_out, activation=activation, weights_init=weights_init,
                                   bias_init=bias_initializer, weight_decay=0.0, name=name)


def conv_transpose(incoming, nb_filter, size, stride, activation=tf.nn.elu):
    _, h, w, n_in = incoming.get_shape().as_list()
    #print(h,w,n_in)
    d = 1 / np.sqrt(size * size * n_in)
    W = tf.Variable(tf.random_uniform([size, size, nb_filter, n_in], minval=-d, maxval=d))
    b = tf.Variable(tf.random_uniform([nb_filter], minval=-d, maxval=d))

    batch_size = tf.gather(tf.shape(incoming), tf.constant([0]))

    output_shape = [h * stride, w * stride, nb_filter]
    complete_out_shape = tf.concat(0, [batch_size, tf.constant(output_shape)])

    conv = tf.nn.conv2d_transpose(incoming, W, complete_out_shape, strides=[1, stride, stride, 1], padding='SAME')
    conv.set_shape([None] + output_shape)
    out = activation(tf.nn.bias_add(conv, b))
    return out


def spatialsoftmax(incoming, epsilon=0.01, trainable_temperature=True, name='SpatialSoftmax', hierarchical=False,
                   safe_softmax=True, use_softmax_only=False, temp_init=0.05):
    # Get the incoming dimensions (should be a 4D tensor)
    _, h, w, c = incoming.get_shape().as_list()

    edge = 1 #1 / (c*2) if not hierarchical else 1 / (c*6)

    with tf.name_scope(name):
        # First we create a linspace from -1 + epsilon to 1 - epsilon. Epsilon is needed to ensure that the output is
        # actually in the range (-1, 1) and not greater than 1 or smaller than -1.
        #
        # Note that each '1' in the reshape is to enforce broadcasting along that dimension
        cartesian_y = tf.reshape(tf.linspace(-edge + epsilon*edge, edge - epsilon*edge, h), (1, h, 1, 1), name="CartesianY")
        cartesian_x = tf.reshape(tf.linspace(-edge + epsilon*edge, edge - epsilon*edge, w), (1, 1, w, 1), name="CartesianX")
        temperature = tf.Variable(initial_value=tf.ones(c) * temp_init, dtype=tf.float32,
                                  trainable=trainable_temperature, name='SoftmaxTemperature')

        # Compute the softmax numerator
        if safe_softmax:
            incoming_ = incoming - tf.stop_gradient(tf.reduce_max(incoming, reduction_indices=[1, 2], keep_dims=True))
            numerator_softmax = tf.exp(incoming_ / tf.reshape(temperature, (1, 1, 1, c)), name='Numerator')
        else:
            numerator_softmax = tf.exp(incoming / tf.reshape(temperature, (1, 1, 1, c)), name='Numerator')
        # The denominator is computed by computing the sum per channel
        # Again, the '1's in the reshaping are to ensure broadcasting along those dimensions
        denominator_softmax = tf.reshape(tf.reduce_sum(numerator_softmax, reduction_indices=[1, 2]), (-1, 1, 1, c),
                                         name='Denominator')
        # Now we compute the softmax per channel
        softmax_per_channel = tf.div(numerator_softmax, denominator_softmax, name='SoftmaxPerChannel')

        if use_softmax_only:
            return tf.reshape(softmax_per_channel, (-1, h * w * c))

        # Compute the x coordinates by element-wise multiplicatoin of the cartesion coordinates with the softmax
        # activations and summing the result
        x_coordinates = tf.reduce_sum(tf.mul(cartesian_x, softmax_per_channel), reduction_indices=[1, 2], name='xOut')
        y_coordinates = tf.reduce_sum(tf.mul(cartesian_y, softmax_per_channel), reduction_indices=[1, 2], name='yOut')

        if hierarchical:
            temperature_patch = tf.Variable(initial_value=tf.ones(c * 4) * temp_init, dtype=tf.float32,
                                            trainable=trainable_temperature, name='SoftmaxTemperaturePatch')
            patched = tf.concat(3, [
                incoming[:,        :h//2,          :w//2,     :],
                incoming[:,    h//2:2*(h//2),      :w//2,     :],
                incoming[:,    h//2:2*(h//2),  w//2:2*(w//2), :],
                incoming[:,        :h//2,      w//2:2*(w//2), :]
            ])
            if safe_softmax:
                patched -= tf.stop_gradient(
                    tf.reduce_max(patched, reduction_indices=[1, 2], keep_dims=True))

            patched /= tf.reshape(temperature_patch, (1, 1, 1, 4*c))

            denominator_softmax = tf.reshape(tf.reduce_sum(patched, reduction_indices=[1, 2]), (-1, 1, 1, c * 4))
            softmax_per_channel = tf.div(patched, denominator_softmax)

            cartesian_y = tf.reshape(tf.linspace(-edge + epsilon*edge, edge - epsilon*edge, h//2), (1, h//2, 1, 1))
            cartesian_x = tf.reshape(tf.linspace(-edge + epsilon*edge, edge - epsilon*edge, w//2), (1, 1, w//2, 1))

            x_coordinates_s = tf.reduce_sum(tf.mul(cartesian_x, softmax_per_channel), reduction_indices=[1, 2],
                                            name='xOut_s')
            y_coordinates_s = tf.reduce_sum(tf.mul(cartesian_y, softmax_per_channel), reduction_indices=[1, 2],
                                            name='yOut_s')

        # Concatenate the resulting tensors to get the output
        o = tf.concat(1, [x_coordinates, y_coordinates] + ([x_coordinates_s, y_coordinates_s] if hierarchical else []),
                      name="Output")
        if trainable_temperature:
            o.b = [temperature] if not hierarchical else [temperature, temperature_patch]
            [tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name + var.name, var) for var in o.b]
        o.sm = softmax_per_channel

    return o


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
                           distance_fn='EXP', padding='same', sigma_trainable=None, per_feature=False,
                           color_coding=False):
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
        :param per_feature      If True, the centroids are given per output feature.
        :param color_coding     If True, uses color coding for visual summary
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

    if color_coding:
        assert incoming.get_shape().as_list()[-1] == 1, "Color coding is only supported for singleton input features"
        assert np.prod(n_centroids) <= 9, "Color coding is only supported for n_centroids <= 9"

        try:
            import colorlover as cl
        except ImportError:
            print("WARNING: Unable to import colorlover, you can install it through 'pip install colorlover --user'\n"
                  "For now, this layer does not use color coding")
            color_coding = False


    with vscope as scope:
        name = scope.name
        with tf.name_scope("SubConvolutions"):
            convs = tflearn.conv_2d(incoming, nb_filter=n_filters * np.prod(n_centroids), filter_size=filter_size,
                                     strides=strides, padding=padding, weight_decay=0., name='Conv',
                                     activation='linear')
            stacked_convs = tf.reshape(convs, [-1] + convs.get_shape().as_list()[1:-1] + [n_filters, np.prod(n_centroids)],
                                       name='StackedConvs')

        _, m, n, k, _ = stacked_convs.get_shape().as_list()

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

            centroids_f = 1 if not per_feature else n_filters

            if isinstance(n_centroids, list):
                assert len(n_centroids) == 2, "Length of n_centroids list must be 2."
                start_x, end_x = scaling / (1 + n_centroids[0]), scaling - scaling / (1 + n_centroids[0])
                start_y, end_y = scaling / (1 + n_centroids[1]), scaling - scaling / (1 + n_centroids[1])

                x_, y_ = tf.meshgrid(tf.linspace(start_x, end_x, n_centroids[0]),
                                     tf.linspace(start_y, end_y, n_centroids[1]))

                centroids_x = tf.Variable(tf.tile(tf.reshape(x_, [np.prod(n_centroids)]), [centroids_f]))
                centroids_y = tf.Variable(tf.tile(tf.reshape(y_, [np.prod(n_centroids)]), [centroids_f]))
                #centroids_y = tf.reshape(tf.transpose(tf.reshape(centroids_y, n_centroids)), [np.prod(n_centroids)])
                n_centroids = np.prod(n_centroids)

            elif isinstance(n_centroids, int):
                centroids_x = tf.Variable(tf.random_uniform([centroids_f * n_centroids], minval=0, maxval=scaling),
                                          trainable=centroids_trainable)
                centroids_y = tf.Variable(tf.random_uniform([centroids_f * n_centroids], minval=0, maxval=scaling),
                                          trainable=centroids_trainable)
            else:
                raise TypeError("n_centroids is neither a list nor an int!")

            #centroids_x = tf.Variable(np.asarray([.5, .5], dtype='float32'), trainable=centroids_trainable)
            #centroids_y = tf.Variable(np.asarray([0, 1], dtype='float32'), trainable=centroids_trainable)

            # Define the distance of each cell w.r.t. the centroids. We can easily accomplish this through broadcasting
            # i.e. x_diff2 will have shape [1, 1, n, 1, c] with n as above and c the number of centroids. Similarly,
            # y_diff2 will have shape [1, m, 1, 1, c]
            x_diff2 = tf.square(tf.reshape(centroids_x, [1, 1, 1, centroids_f, n_centroids]) - x_coordinates,
                                name='xDiffSquared')
            y_diff2 = tf.square(tf.reshape(centroids_y, [1, 1, 1, centroids_f, n_centroids]) - y_coordinates,
                                name='yDiffSquared')

            if distance_fn == 'EXP':
                sigma = tf.Variable(scaling/2 * np.ones(n_centroids * centroids_f, dtype='float'), dtype=tf.float32,
                                    name='Sigma', trainable=sigma_trainable)

            # Again, we use broadcasting. The result is of shape [1, m, n, 1, c]
            euclidian_dist = np.sqrt(2) - tf.sqrt(x_diff2 + y_diff2, name="Euclidean") if distance_fn == 'EUCLIDEAN' \
                else tf.exp(-tf.div((x_diff2 + y_diff2),
                                    tf.reshape(sigma ** 2, [1, 1, 1, centroids_f, n_centroids])), 'Exp')

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
            out.W = [convs.W]
            out.b = tf.concat(0, [convs.b] + [centroids_x, centroids_y] +
                              ([sigma] if distance_fn == 'EXP' and sigma_trainable else []),
                              name='b_concatAndCentroids')

            # Add to collection for tflearn functionality
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, out.W)
            tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + name, out.b)
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS + '/' + name, out)

        with tf.name_scope("Summary"):
            euclidean_downsampled = euclidian_dist[:, ::3, ::3, :, :]
            _, m, n, _, _ = euclidean_downsampled.get_shape().as_list()

            if per_feature:
                # In case we have centroids per feature, we need to make sure that the centroids dimension is at the 2nd
                # axis
                euclidean_downsampled = tf.transpose(euclidean_downsampled, [0, 3, 1, 2, 4])

            # The euclidean_downsampled Tensor contains the spatial coefficients for different centroids
            # We can reshape this such that we multiply the corresponding filters with the help of broadcasting
            euclidian_dist_reshaped = tf.reshape(euclidean_downsampled, 3 * [1] + [centroids_f, m, n, n_centroids])
            current_filter_shape = convs.W.get_shape().as_list()
            current_filter_shape[-1] //= n_centroids

            # Now we stack the weights, that is, there is an extra axis for the centroids
            weights_stacked = tf.reshape(convs.W, current_filter_shape + [1, 1, n_centroids])

            # Get the locally weighted kernels
            locally_weighted_kernels = tf.reduce_sum(tf.mul(euclidian_dist_reshaped, weights_stacked), axis=6)

            if color_coding:
                colors_numeric = [list(c) for c in cl.to_numeric(cl.scales['9']['qual']['Set1'][:n_centroids])]
                colors = tf.reshape(colors_numeric, (1, 1, 3, 1, 1, 1, n_centroids))
                color_distance = tf.tile(tf.reduce_sum(
                    euclidian_dist_reshaped * colors, axis=6), current_filter_shape[:-1] + 3*[1])
                locally_weighted_kernels = tf.concat(2, [color_distance, 1. - locally_weighted_kernels])

                current_filter_shape[2] = 4


            # Normalize
            locally_weighted_kernels -= tf.reduce_min(locally_weighted_kernels, axis=[4, 5], keep_dims=True)
            locally_weighted_kernels /= tf.reduce_max(locally_weighted_kernels, axis=[4, 5], keep_dims=True)

            # Now comes the tricky part to get all the locally weighted kernels grouped in a tiled image. We need to to
            # do quite some transposing. First we transpose the locally weighted kernels such that the first two axes
            # correspond to the #in and #out channels, the 3rd and 4th correspond to rows and columns of the kernels,
            # the last two dimensions correspond to the spatial locations of the kernels.
            in_out_kernel_spatial = tf.transpose(locally_weighted_kernels, [2, 3, 0, 1, 4, 5])

            # Now we flatten the last two dimensions, effectively taking the images for the spatial locations on a
            # single row. We also apply some padding, so that we can visually separate the kernels easily
            in_out_kernel_spatial_flattened = tf.pad(tf.reshape(in_out_kernel_spatial, [current_filter_shape[2],
                                                                                        current_filter_shape[3],
                                                                                        current_filter_shape[0],
                                                                                        current_filter_shape[1],
                                                                                        m * n], name='Flattening'),
                                                     [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]])
            current_filter_shape[0] += 1
            current_filter_shape[1] += 1

            # Transpose again, again we first have axes for in and out channels, followed by the flattened spatial
            # locations, and finally by the row and column axes of the kernels themselves
            in_out_spatial_f_kernel = tf.transpose(in_out_kernel_spatial_flattened, [0, 1, 4, 2, 3])

            # Now we take together the spatial rows and filter rows
            in_out_y_reshaped = tf.reshape(in_out_spatial_f_kernel, [current_filter_shape[2],
                                                                     current_filter_shape[3],
                                                                     n,
                                                                     m * current_filter_shape[0],
                                                                     current_filter_shape[1]])

            # Now we do the same for combining the columns
            in_out_y_switched = tf.transpose(in_out_y_reshaped, [0, 1, 2, 4, 3])
            in_out_grid = tf.reshape(in_out_y_switched, [current_filter_shape[2],
                                                         current_filter_shape[3],
                                                         n * current_filter_shape[1],
                                                         m * current_filter_shape[0]])

            # And we are done! We want the last dimension to be the depth of the images we display, so it makes sense
            # to transpose it one last time. Remember that for our rows were put at the end, so that they need to be on
            # the 2nd axis now
            summary_image = tf.transpose(in_out_grid, [1, 3, 2, 0])

            out.visual_summary = summary_image
            out.W_list = tf.split(3, n_centroids, convs.W)

    # Add to collection for tflearn functionality
    tf.add_to_collection(tf.GraphKeys.LAYER_TENSOR + '/' + name, out)
    return out


