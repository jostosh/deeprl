import tensorflow as tf
import numpy as np
from deeprl.common.logger import logger


def layer(fn):
    """
    Provides wrapper for layer methods that will call _finalize after creation and will make sure 'incoming' in kwargs
    is either the last added layer or it is provided explicitly through a keyword argument
    """
    from functools import wraps

    @wraps(fn)
    def layer_wrapper(self, *args, **kwargs):
        if 'incoming' not in kwargs:
            kwargs['incoming'] = self.head
        out = fn(self, *args, **kwargs)
        self._finalize(out, kwargs.get('name'))
        return out
    return layer_wrapper


class DNN:

    def __init__(self, input=None):
        self.head = input
        self.theta = []
        self.outputs = {}

    @layer
    def conv_layer(self, n_filters, filter_size, stride, activation, name, padding='VALID', init='torch',
                   incoming=None):
        """ Implements a convolutional layer """
        _, kh, kw, input_channels = incoming.get_shape().as_list()
        with tf.variable_scope(name):
            bias_init, weight_init = self._get_initializers(init, input_channels, filter_size)
            W = tf.get_variable('W', shape=[filter_size, filter_size, input_channels, n_filters],
                                initializer=weight_init)
            b = tf.get_variable('b', shape=[n_filters], initializer=bias_init)

            self._add_theta(W, b)
            out = activation(tf.nn.bias_add(tf.nn.conv2d(incoming, W, [1, stride, stride, 1], padding), b))

        return out

    @layer
    def fc_layer(self, n_out, activation, name, init='torch', incoming: tf.Tensor=None):
        """ Implements a fully connected layer """
        if incoming.get_shape().ndims == 4:
            incoming = self.flatten(incoming=incoming, name="Flatten")
        n_in = incoming.get_shape().as_list()[-1]

        with tf.variable_scope(name):
            weight_init, bias_init = self._get_initializers(init, n_in)
            W = tf.get_variable('W', shape=[n_in, n_out], initializer=weight_init)
            b = tf.get_variable('b', shape=[n_out],       initializer=bias_init)
            self._add_theta(W, b)
            out = activation(tf.nn.xw_plus_b(incoming, W, b))

        return out

    @layer
    def spatial_softmax(self, epsilon=0.5, trainable_temperature=True, name='SpatialSoftmax', incoming=None):
        """
        Implements spatial softmax operator which can be found at https://arxiv.org/abs/1504.00702
        Incoming should be a 4D tensor.
        """
        # Get the incoming dimensions (should be a_t 4D tensor)
        _, h, w, c = incoming.get_shape().as_list()

        edge = 1

        with tf.variable_scope(name):
            # First we create a_t linspace from -1 + epsilon to 1 - epsilon.
            # Epsilon is needed to ensure that the output is actually in the range (-1, 1) and not greater than 1
            # or smaller than -1.
            #
            # Note that each '1' in the reshape is to enforce broadcasting along that dimension
            cartesian_y = tf.reshape(tf.linspace(-edge + epsilon * edge, edge - epsilon * edge, h), (1, h, 1, 1),
                                     name="CartesianY")
            cartesian_x = tf.reshape(tf.linspace(-edge + epsilon * edge, edge - epsilon * edge, w), (1, 1, w, 1),
                                     name="CartesianX")
            temperature = tf.Variable(initial_value=tf.ones(c) / 10, dtype=tf.float32, trainable=trainable_temperature)
            if trainable_temperature:
                self._add_theta(temperature)

            # Compute the softmax numerator
            numerator_softmax = tf.exp(incoming / tf.reshape(temperature, (1, 1, 1, c)), name='Numerator')
            # The denominator is computed by computing the sum per channel
            # Again, the '1's_t in the reshaping are to ensure broadcasting along those dimensions
            denominator_softmax = tf.reshape(tf.reduce_sum(numerator_softmax, reduction_indices=[1, 2]), (-1, 1, 1, c),
                                             name='Denominator')
            # Now we compute the softmax per channel
            softmax_per_channel = tf.div(numerator_softmax, denominator_softmax, name='SoftmaxPerChannel')

            # Compute the x coordinates by element-wise multiplication of the Cartesian coordinates with the softmax
            # activations and sums the result to extract x-coordinates and y-coordinates.
            x_coordinates = tf.reduce_sum(tf.mul(cartesian_x, softmax_per_channel), reduction_indices=[1, 2],
                                          name='xOut')
            y_coordinates = tf.reduce_sum(tf.mul(cartesian_y, softmax_per_channel), reduction_indices=[1, 2],
                                          name='yOut')

            # Concatenate the resulting tensors to get the output
            out = tf.concat(1, [x_coordinates, y_coordinates], name="Output")

        return out

    @layer
    def lpq_layer(self, ppa, n_classes, name, sim_fn, glpq=False, temperature=1.0, incoming=None):
        """
        Implements LPQ layer
        Parameters:
            :param ppa:         Prototypes per action
            :param n_classes:   Number of classes
            :param name:        Name used in variable scope
            :param sim_fn       Similarity function
            :param glpq         Whether to use GLPQ. If true will compute relative similarity
            :param temperature  Softmax temperature
            :param incoming     Incoming tensor
        Returns:
            :return A Tensor with dimensions [batch_size, n_classes] representing the LPQ output
        """
        # Prototype initlialization
        num_prototypes = ppa * n_classes
        n_in = incoming.get_shape().as_list()[-1]
        d = 1.0 / np.sqrt(n_in)
        prototype_init = tf.nn.relu(tf.random_uniform((num_prototypes, n_in), minval=-d, maxval=d))

        with tf.variable_scope(name):
            # Add prototypes to variables
            prototypes = tf.Variable(prototype_init, name='Prototypes')
            self._add_theta(prototypes)

            # Compute the similarity score
            similarity = lpq_similarity(incoming, prototypes, sim_fn)
            if glpq:
                # Compute relative similarity
                similarity = tf.reshape(similarity, [-1, n_classes,  ppa])
                similarity = relative_similarity(similarity, n_classes)
                similarity = tf.reshape(similarity, [-1, n_classes * ppa])

            similarity *= temperature                                                          # Multiply by temperature
            prototype_scores = tf.nn.softmax(similarity)                                       # Softmax per prototype
            out = tf.reduce_sum(tf.reshape(prototype_scores, [-1, n_classes, ppa]), axis=2)    # Sum per class

        return out

    @layer
    def local_weight_sharing(self, n_centroids, n_filters, filter_size, strides, activation,
                             name='SoftWeightConv',
                             scope=None, reuse=False, local_normalization=True, centroids_trainable=False,
                             scaling=1.0, incoming=None,
                             padding='VALID', sigma_trainable=None, per_feature=False, color_coding=False,
                             similarity_fn='Exp', init='torch'):
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
            :param padding          Padding to use for the sub-convolutions. It is important to have the centroids aligned
                                    for stacked spatial weight sharing layers
            :param sigma_trainable  Whether the sigma parameters of the exponential distance function should be trainable.
                        By default, this parameter is set to None, in which case it takes over the value of
                        centroids_trainable.
            :param per_feature      If True, the centroids are given per output feature.
            :param color_coding     If True, uses color coding for visual summary
            :param similarity_fn    Similarity function to be used. Can be either 'Exp' or 'InvEuclidean'
        Return values:
            :return: A 4D Tensor with similar dimensionality as a normal conv_2d's output
        """
        valid_fns = ['Exp', 'InvEuclidean']
        if similarity_fn not in valid_fns:
            raise ValueError("Invalid similarity function {}, must be either {}"
                             .format(similarity_fn, ' or '.join(valid_fns)))
        try:
            vscope = tf.variable_scope(scope, default_name=name, values=[incoming],
                                       reuse=reuse)
        except Exception:
            vscope = tf.variable_op_scope([incoming], scope, name, reuse=reuse)

        if sigma_trainable is None:
            sigma_trainable = centroids_trainable

        if color_coding:
            assert incoming.get_shape().as_list()[
                       -1] == 1, "Color coding is only supported for singleton input features"
            assert np.prod(n_centroids) <= 12, "Color coding is only supported for n_centroids <= 12"

            try:
                import colorlover as cl
            except ImportError:
                logger.WARN("Unable to import colorlover, you can install it through 'pip install colorlover --user'\n" \
                            "For now, this layer does not use color coding.")
                color_coding = False

        with vscope as scope:
            name = scope.name
            with tf.name_scope("SubConvolutions"):
                input_channels = incoming.get_shape().as_list()[-1]
                weight_init, bias_init = self._get_initializers(init, input_channels, filter_size)
                W = tf.get_variable('W', shape=[filter_size, filter_size, input_channels,
                                                n_filters * np.prod(n_centroids)])
                b = tf.get_variable('b', shape=[n_filters * np.prod(n_centroids)])
                convs = tf.nn.bias_add(tf.nn.conv2d(incoming, W, strides=[1, strides, strides, 1], padding=padding), b)
                stacked_convs = tf.reshape(convs, [-1] + convs.get_shape().as_list()[1:-1] +
                                           [n_filters, np.prod(n_centroids)], name='StackedConvs')
                _, m, n, k, _ = stacked_convs.get_shape().as_list()

            with tf.name_scope("DistanceWeighting"):
                # First define the x-coordinates per cell. We exploit TensorFlow's broadcast mechanisms by using
                # single-sized dimensions
                y_coordinates = tf.reshape(tf.linspace(0., scaling, m), [1, m, 1, 1, 1])
                x_coordinates = tf.reshape(tf.linspace(0., scaling, n), [1, 1, n, 1, 1])

                # This is a dimension size needed for feature-wise configuration of centroids
                centroids_f = 1 if not per_feature else n_filters

                # Define the centroids variables
                if isinstance(n_centroids, list):
                    # In case n_centroids is a list, we initialize the centroids in a grid
                    assert len(n_centroids) == 2, "Length of n_centroids list must be 2."

                    # If we initialize the centroids in a grid and take them to be non-trainable, it does not make
                    # sense to have feature-wise KCPs
                    if not centroids_trainable:
                        centroids_f = 1
                        per_feature = False

                    start_x, end_x = scaling / (1 + n_centroids[1]), scaling - scaling / (1 + n_centroids[1])
                    start_y, end_y = scaling / (1 + n_centroids[0]), scaling - scaling / (1 + n_centroids[0])

                    x_, y_ = tf.meshgrid(tf.linspace(start_x, end_x, n_centroids[1]),
                                         tf.linspace(start_y, end_y, n_centroids[0]))

                    centroids_x = tf.Variable(tf.reshape(tf.tile(
                        tf.reshape(x_, [1, np.prod(n_centroids)]), [centroids_f, 1]), [-1]),
                        trainable=centroids_trainable)
                    centroids_y = tf.Variable(tf.reshape(tf.tile(
                        tf.reshape(y_, [1, np.prod(n_centroids)]), [centroids_f, 1]), [-1]),
                        trainable=centroids_trainable)
                    n_centroids = np.prod(n_centroids)

                elif isinstance(n_centroids, int):
                    # In case n_centroids is an int, we initialize the centroids randomly
                    centroids_x = tf.Variable(tf.random_uniform([centroids_f * n_centroids], minval=0, maxval=scaling),
                                              trainable=centroids_trainable)
                    centroids_y = tf.Variable(tf.random_uniform([centroids_f * n_centroids], minval=0, maxval=scaling),
                                              trainable=centroids_trainable)
                else:
                    raise TypeError("n_centroids is neither a list nor an int!")

                # Define the distance of each cell w.r.t. the centroids. We can easily accomplish this through
                # broadcasting i.e. x_diff2 will have shape [1, 1, n, 1, c] with n as above and c the number of
                # centroids. Similarly, y_diff2 will have shape [1, m, 1, 1, c]

                centroid_broadcasting_shape = [1, 1, 1, centroids_f, n_centroids]

                cov11 = tf.Variable(scaling * np.ones(n_centroids * centroids_f, dtype='float'), dtype=tf.float32,
                                    name='Sigma11', trainable=sigma_trainable)
                cov12 = tf.Variable(scaling * np.zeros(n_centroids * centroids_f, dtype='float'), dtype=tf.float32,
                                    name='Sigma12', trainable=sigma_trainable)
                cov21 = tf.Variable(scaling * np.zeros(n_centroids * centroids_f, dtype='float'), dtype=tf.float32,
                                    name='Sigma21', trainable=sigma_trainable)
                cov22 = tf.Variable(scaling * np.ones(n_centroids * centroids_f, dtype='float'), dtype=tf.float32,
                                    name='Sigma22', trainable=sigma_trainable)

                x_diff = tf.reshape(centroids_x, centroid_broadcasting_shape) - x_coordinates
                y_diff = tf.reshape(centroids_y, centroid_broadcasting_shape) - y_coordinates

                x_diff2 = tf.square(x_diff * tf.reshape(cov11, centroid_broadcasting_shape) +
                                    y_diff * tf.reshape(cov12, centroid_broadcasting_shape))
                y_diff2 = tf.square(x_diff * tf.reshape(cov21, centroid_broadcasting_shape) +
                                    y_diff * tf.reshape(cov22, centroid_broadcasting_shape))

                # Again, we use broadcasting. The result is of shape [1, m, n, 1, c]
                similarities = tf.exp(-(x_diff2 + y_diff2), name='similarities')

                # Optionally, we will perform local normalization such that the weight coefficients add up to 1 for each
                # spatial cell.
                if local_normalization:
                    # Add up the similarities locally
                    total_distance_per_cell = tf.reduce_sum(similarities, axis=4, keep_dims=True)
                    # Now divide
                    similarities = tf.div(similarities, total_distance_per_cell, name='NormalizedSimilarity')

            with tf.name_scope("SoftWeightSharing"):
                # Compute the distance-weighted output
                dist_weighted = tf.multiply(similarities, stacked_convs, name='DistanceWeighted')

                # Apply non-linearity
                out = activation(tf.reduce_sum(dist_weighted, axis=4), name='Output')

                self._add_theta(convs.W, convs.b)
                if centroids_trainable:
                    self._add_theta(centroids_x, centroids_y)
                if sigma_trainable:
                    self._add_theta(cov11, cov12, cov21, cov22)

            with tf.name_scope("Summary"):
                summary_image = build_visualization(centroids_f, color_coding, convs, n_centroids, n_filters,
                                                    per_feature, similarities)
                out.visual_summary = summary_image
                try:
                    out.W_list = tf.split(3, n_centroids, convs.W)
                except:
                    out.W_list = tf.split(convs.W, n_centroids * [n_filters], axis=3)

        return out

    @layer
    def flatten(self, _guard=None, incoming=None, name='Flatten'):
        """ Flattens input to a Tensor of rank 2 """
        assert _guard is None, "DNN.flatten: only provide kwargs!"
        n_out = np.prod(incoming.get_shape().as_list()[1:])
        out = tf.reshape(incoming, [-1, n_out])
        return out

    def _finalize(self, out, name):
        self.head = out
        self.outputs[name] = out
        return out

    @staticmethod
    def _get_initializers(init, input_channels, filter_size=1):
        """ Weight initializers """
        if init == 'torch':
            # This is used in Deepmind's papers
            d = 1.0 / np.sqrt(filter_size * filter_size * input_channels)
            weight_init = tf.random_uniform_initializer(minval=-d, maxval=d)
            bias_init = tf.random_uniform_initializer(minval=-d, maxval=d)
        else:
            weight_init = tf.contrib.layers.variance_scaling_initializer()
            bias_init = tf.constant_initializer(0.0)
        return bias_init, weight_init

    def _add_theta(self, *args):
        self.theta.extend(list(args))



def either(a, b):
    """ This allows a tf.Tensor compatible version of 'var = a or b' """
    return b if a is None else a


def lpq_similarity(net, prototypes, fn_name):
    """ Currently we only implement two similarity functions """
    return {
        'euc2': euclidean_squared,
        'man':  manhattan
    }[fn_name](net, prototypes)


def euclidean_squared(net, prototypes):
    """ Squared Euclidean similarity """
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.square(diff), axis=2)


def manhattan(net, prototypes):
    """ Manhattan similarity """
    diff = tf.expand_dims(net, 1) - tf.expand_dims(prototypes, 0)
    return -tf.reduce_sum(tf.abs(diff), axis=2)


def relative_similarity(similarities, num_classes):
    """ Computes relative similarity used for GLPQ """
    scores = []
    _, na, num_p = similarities.get_shape().as_list()
    for i in range(num_classes):
        # Get indices j != i
        j_neq_i = [j for j in range(num_classes) if j != i]

        # Get the most similar other prototype belonging to another class j != i
        similarities_other = tf.reshape(tf.reduce_max(
            tf.transpose(tf.gather(tf.transpose(similarities, (1, 0, 2)), j_neq_i), (1, 0, 2)),
            axis=[1, 2]),
            shape=(-1, 1)
        )

        # Get the similarities of the prototypes belonging to class i
        similarities_same = similarities[:, i, :]

        # Compute relative similarity
        scores.append((similarities_same - similarities_other) / (similarities_same + similarities_other))

    return tf.stack(scores, axis=1)


def color_augmentation(current_filter_shape, locally_weighted_kernels, n_centroids, n_filters, per_feature,
                       similarities):
    """ Function to visualize local weight sharing """
    import colorlover as cl
    # Add color coding such that each centroid codes for a certain color.
    colors_numeric = [
        list(c) for c in cl.to_numeric(cl.scales['9']['qual']['Set1'])[:n_centroids]
    ]
    colors = tf.reshape(tf.transpose(colors_numeric), (1, 1, 3, 1, 1, 1, n_centroids)) / 255.
    # We set all colors of non-maximum centroid similarity to zero. This is to avoid the blending of colors
    # that impedes the visual separability of the kernels
    max_indices = tf.argmax(tf.reshape(similarities, [-1, n_centroids]), axis=1)
    max_mask = tf.reshape(tf.one_hot(max_indices, depth=n_centroids, on_value=1., off_value=0.),
                          similarities.get_shape())
    # The colored distances are given by multiplying the similarities with the color values, and again, we
    # exploit the use of broadcasting
    colored_distances = similarities * colors * max_mask
    color_max = tf.reduce_max(colored_distances, axis=[2, 4, 5], keep_dims=True)
    color_min = tf.reduce_min(colored_distances, axis=[2, 4, 5], keep_dims=True)
    # Now we can compute
    color_aug = tf.clip_by_value(
        tf.tile(
            tf.reduce_sum((colored_distances - color_min) / (color_max - color_min + 1e-8), axis=6),
            current_filter_shape[:-1] + [1 if per_feature else n_filters, 1, 1]), 0., 0.999)
    locally_weighted_kernels = tf.concat(2, [color_aug, locally_weighted_kernels])
    current_filter_shape[2] = 4
    return locally_weighted_kernels


def build_visualization(centroids_f, color_coding, conv_layer, n_centroids, n_filters, per_feature, similarities):
    similarities_downsampled = similarities[:, ::3, ::3, :, :]
    _, m, n, _, _ = similarities_downsampled.get_shape().as_list()
    if per_feature:
        # In case we have centroids per feature, we need to make sure that the centroids dimension is at the 2nd
        # axis
        similarities_downsampled = tf.transpose(similarities_downsampled, [0, 3, 1, 2, 4])

    # The euclidean_downsampled Tensor contains the spatial coefficients for different centroids
    # We can reshape this such that we multiply the corresponding filters with the help of broadcasting
    similarities_downsampled = tf.reshape(similarities_downsampled, 3 * [1] + [centroids_f, m, n, n_centroids])
    filter_shape = conv_layer.W.get_shape().as_list()
    filter_shape[-1] //= n_centroids
    # Now we stack the weights, that is, there is an extra axis for the centroids
    weights_stacked = tf.reshape(conv_layer.W, filter_shape + [1, 1, n_centroids])
    # Get the locally weighted kernels
    locally_weighted_kernels = tf.reduce_sum(tf.multiply(similarities_downsampled, weights_stacked), axis=6)
    # Normalize
    locally_weighted_kernels -= tf.reduce_min(locally_weighted_kernels, axis=[4, 5], keep_dims=True)
    locally_weighted_kernels /= tf.reduce_max(locally_weighted_kernels, axis=[4, 5], keep_dims=True)
    if color_coding:
        locally_weighted_kernels = color_augmentation(filter_shape, locally_weighted_kernels, n_centroids,
                                                      n_filters, per_feature, similarities_downsampled)

    # Now comes the tricky part: getting all the locally weighted kernels grouped in a tiled image. We need to to
    # do quite some transposing and reshaping. First we transpose the locally weighted kernels such that the first two
    # axes correspond to the #in and #out channels, the 3rd and 4th correspond to rows and columns of the kernels,
    # the last two dimensions correspond to the spatial locations of the kernels.
    in_out_kernel_spatial = tf.transpose(locally_weighted_kernels, [2, 3, 0, 1, 4, 5])

    # Now we flatten the last two dimensions, effectively taking the images for the spatial locations on a
    # single row. We also apply some padding, so that we can visually separate the kernels easily
    in_out_kernel_spatial_flattened = tf.pad(tf.reshape(in_out_kernel_spatial, [filter_shape[2],
                                                                                filter_shape[3],
                                                                                filter_shape[0],
                                                                                filter_shape[1],
                                                                                m * n], name='Flattening'),
                                             [[0, 0], [0, 0], [1, 0], [1, 0], [0, 0]])
    filter_shape[0] += 1
    filter_shape[1] += 1

    # Transpose again, again we first have axes for in and out channels, followed by the flattened spatial
    # locations, and finally by the row and column axes of the kernels themselves
    in_out_spatial_f_kernel = tf.transpose(in_out_kernel_spatial_flattened, [0, 1, 4, 2, 3])

    # Now we take together the spatial rows and filter rows
    in_out_y_reshaped = tf.reshape(in_out_spatial_f_kernel, [filter_shape[2],
                                                             filter_shape[3],
                                                             n,
                                                             m * filter_shape[0],
                                                             filter_shape[1]])

    # Now we do the same for combining the columns
    in_out_y_switched = tf.transpose(in_out_y_reshaped, [0, 1, 2, 4, 3])
    in_out_grid = tf.reshape(in_out_y_switched, [filter_shape[2],
                                                 filter_shape[3],
                                                 n * filter_shape[1],
                                                 m * filter_shape[0]])

    # And we are done! We want the last dimension to be the depth of the images we display, so it makes sense
    # to transpose it one last time. Remember that our rows were put at the end, so they need to be on the
    # 2nd axis now. To make it look even neater, we do some additional padding around the edges
    summary_image = tf.pad(tf.transpose(in_out_grid, [1, 3, 2, 0]), paddings=[[0, 0], [0, 1], [0, 1], [0, 0]])
    return summary_image

