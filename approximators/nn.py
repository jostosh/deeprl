import tensorflow as tf
import tflearn
import numpy as np
from deeprl.common.logger import logger
from deeprl.approximators.layers import spatialsoftmax, custom_lstm, convolutional_lstm, conv_layer, fc_layer
from deeprl.approximators.sisws import spatial_weight_sharing
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from copy import deepcopy
from deeprl.approximators.convlstm import ConvLSTM2D

class ModelNames:
    A3C_FF      = 'a3c_ff'
    NATURE      = 'nature'
    SMALL_FCN   = 'small_fcn'
    A3C_LSTM    = 'a3c_lstm'
    A3C_FF_SS   = 'a3c_ff_ss'
    A3C_LSTM_SS = 'a3c_lstm_ss'
    A3C_CONV_LSTM = 'a3c_conv_lstm'
    A3C_CONV_LSTM_K = 'a3c_conv_lstm_k'
    A3C_SISWS   = 'a3c_sisws'
    A3C_SISWS_S = 'a3c_sisws_s'



class ActorCriticNN(object):

    def __init__(self, num_actions, agent_name, optimizer, hyper_parameters, global_network=None):
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_network = global_network
        #self.session = session
        self.beta = hyper_parameters.beta
        self.summaries = []
        self.agent_name = agent_name
        self.model_name = hyper_parameters.model
        self.clip_advantage = hyper_parameters.clip_advantage
        self.recurrent = self.model_name in [ModelNames.A3C_LSTM, ModelNames.A3C_LSTM_SS, ModelNames.A3C_CONV_LSTM,
                                             ModelNames.A3C_CONV_LSTM_K]
        self.t_max = hyper_parameters.t_max
        self.input_shape = hyper_parameters.input_shape
        self.policy_weighted_val = hyper_parameters.policy_weighted_val
        self.frame_prediction = hyper_parameters.frame_prediction
        self.residual_prediction = hyper_parameters.residual_prediction
        self.layers = {}
        self.embedding_layer = None
        self.gamma = hyper_parameters.gamma
        self.clip_rewards = hyper_parameters.clip_rewards
        self.optimality_tightening = hyper_parameters.optimality_tightening
        self.lstm_state_numeric = self.lstm_first_state_since_update = None
        #self.policy_quanization = hyper_parameters.policy_quantization

        self.hp = hyper_parameters

        # Build computational graphs for loss, synchronization of parameters and parameter updates
        with tf.name_scope(agent_name):
            self.build_network(num_actions, hyper_parameters.input_shape)

            with tf.name_scope('Loss') as loss_scope:
                self.build_loss()

            self.loss_scope = loss_scope

            if self.frame_prediction:
                with tf.name_scope('FramePrediction'):
                    self.build_frame_predictor()

            if global_network and global_network != 'mpi':
                # If this NN has a global copy
                self.build_param_sync()
                self.build_param_update()

            if global_network == 'mpi':
                self.build_param_update()
        self.reset()
        self.merged_summaries = tf.summary.merge(self.summaries)#tf.merge_summary(self.summaries)

    def _nips_hidden_layers(self):
        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = conv_layer(net, 32, 8, 4, activation='linear', name='Conv1')

            self._add_trainable(net)
            net = self.hp.activation(net)

            # Add second convolutional layer
            net = conv_layer(net, 64, 4, 2, activation='linear', name='Conv2')
            self._add_trainable(net)
            net = self.hp.activation(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)
            self.embedding_layer = net

        return net, scope

    def _a3c_sisws(self):
        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = spatial_weight_sharing(net, 3, n_filters=32, filter_size=8, strides=4, activation=self.hp.activation,
                                         name='Conv1', centroids_trainable=True, per_feature=True)
            self._add_trainable(net)

            # Add second convolutional layer
            net = spatial_weight_sharing(net, 3, n_filters=32, filter_size=4, strides=2, activation=self.hp.activation,
                                         name='Conv2', centroids_trainable=True, per_feature=True)
            self._add_trainable(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)

            self.embedding_layer = net

        return net

    def _a3c_sisws_s(self):
        with tf.name_scope(self.forward_input_scope):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers') as scope:
            # Add first convolutional layer
            net = spatial_weight_sharing(net, [2, 2], n_filters=32, filter_size=8, strides=4,
                                         activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)

            # Add second convolutional layer
            net = spatial_weight_sharing(net, [2, 2], n_filters=32, filter_size=4, strides=2,
                                         activation=self.hp.activation, name='Conv2')
            self._add_trainable(net)

            net = tflearn.flatten(net)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)

            self.embedding_layer = net

        return net

    def _nature_model(self):
        """
        This is the more complicated model as taken from the nature paper in 2015.
        :param network_name:   Name of the network
        :return:               The feedforward model (last hidden layer as a graph node)
        """
        with tf.name_scope('ForwardInputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 8, 4, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = conv_layer(net, 64, 4, 2, activation=self.hp.activation, name='Conv2')
            self._add_trainable(net)
            net = conv_layer(net, 64, 3, 1, activation=self.hp.activation, name='Conv3')
            self._add_trainable(net)
            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 512, activation=self.hp.activation, name='FC4')
            self._add_trainable(net)

        return net

    def _a3c_ff(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        return self._nips_hidden_layers()[0]

    def _a3c_ff_ss(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning" together with a
        spatial softmax layer.
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 5, 3, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = conv_layer(net, 64, 3, 2, activation=tf.identity, name='Conv2')
            self._add_trainable(net)
            net = spatialsoftmax(net, epsilon=self.hp.ss_epsilon, trainable_temperature=self.hp.trainable_temp,
                                 safe_softmax=self.hp.trainable_temp, use_softmax_only=self.hp.softmax_only)
            if self.hp.trainable_temp:
                self.theta += net.b
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)
            self.embedding_layer = net

        return net

    def _a3c_lstm(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('LSTMInput'):
            # An LSTM layers's 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.initial_state = LSTMStateTuple(
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c"),
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            )
            self.n_steps = tf.placeholder(tf.int32, shape=[1])

        net, scope = self._nips_hidden_layers()

        with tf.name_scope(scope) as scope:
            net = tflearn.reshape(net, [1, -1, 256], "ReshapedLSTMInput")
            net, state = custom_lstm(net, 256, initial_state=self.initial_state, name='LSTM4_{}'.format(self.agent_name),
                                     sequence_length=self.n_steps)
            self._add_trainable(net)
            net = tflearn.reshape(net, [-1, 256], name="ReshapedLSTMOutput")
            self.lstm_state_variable = state
            self.embedding_layer = net

        return net

    def _a3c_lstm_ss(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a graph node
        """
        with tf.name_scope('LSTMStateInput'):
            # An LSTM layers's 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.initial_state = LSTMStateTuple(
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c"),
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            )
            self.n_steps = tf.placeholder(tf.int32, shape=[1])
        with tf.name_scope('ForwardInputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])
        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 8, 4, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = conv_layer(net, 64, 4, 2, activation='linear', name='Conv2')
            self._add_trainable(net)
            net = spatialsoftmax(net, epsilon=0.99)
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self._add_trainable(net)
            net = tflearn.reshape(net, [1, -1, 256], "ReshapedLSTMInput")
            net, state = custom_lstm(net, 256, initial_state=self.initial_state,
                                     name='LSTM4_{}'.format(self.agent_name),
                                     sequence_length=self.n_steps)
            self._add_trainable(net)
            net = tflearn.reshape(net, [-1, 256], name="ReshapedLSTMOutput")
            self.lstm_state_variable = state
            self.embedding_layer = net
        return net

    def _a3c_conv_lstm(self):
        """
        This is an experimental architecture that uses convolutional LSTM layers.
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('LSTMInput'):
            # An LSTM layers's 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.n_steps = tf.placeholder(tf.int32, shape=[1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 8, 4, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = tf.reshape(net, [-1, 1] + net.get_shape().as_list()[1:])

            net, new_state, self.initial_state = convolutional_lstm(net, outer_filter_size=4, num_features=64,
                                                                    stride=2, inner_filter_size=5, inner_depthwise=False,
                                                                    forget_bias=1.)
            self.theta += net.W + [net.b]

            net = tf.reshape(net, [-1, 9 * 9 * 64])
            self.lstm_state_variable = new_state
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self.embedding_layer = net

        return net


    def _a3c_conv_lstm_k(self):
        """
        This is an experimental architecture that uses convolutional LSTM layers.
        """
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('LSTMInput'):
            # An LSTM layers's 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.n_steps = tf.placeholder(tf.int32, shape=[1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 8, 4, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = tf.reshape(net, [-1, 1] + net.get_shape().as_list()[1:])

            self.initial_state = LSTMStateTuple(c=tf.placeholder(tf.float32, [None, 9, 9, 64]),
                                                h=tf.placeholder(tf.float32, [None, 9, 9, 64]))

            #net, new_state, self.initial_state = convolutional_lstm(net, outer_filter_size=4, num_features=64,
            #                                                        stride=2, inner_filter_size=5, inner_depthwise=False,
            #                                                        forget_bias=1.)
            convlstm_layer = ConvLSTM2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2, 2), dim_ordering='tf',
                                        state_ph=self.initial_state, return_sequences=True, nb_row_i=3, nb_col_i=3,
                                        inner_activation=tf.nn.sigmoid)
            net = convlstm_layer(net)

            self.theta += convlstm_layer.trainable_weights
            self.lstm_state_variable = convlstm_layer.state_out

            net = tf.reshape(net, [-1, 9 * 9 * 64])
            #self.lstm_state_variable = new_state
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self.embedding_layer = net

        return net


    def _small_fcn(self):
        """
        This network works with the CartPole-v0/v1 environments (sort of)
        :param network_name:    The name of the network
        :return:                The network as a graph node
        """
        with tf.name_scope('HiddenLayers'):
            net = fc_layer(self.inputs, 128, activation='tanh', name='FC1')
            self._add_trainable(net)
        self.embedding_layer = net
        return net

    def reset_lstm_state(self):
        if self.lstm_state_numeric is None and self.lstm_first_state_since_update is None:
            if self.model_name == ModelNames.A3C_CONV_LSTM:
                self.lstm_state_numeric = np.zeros(self.initial_state.get_shape().as_list())
                self.lstm_first_state_since_update = np.zeros(self.initial_state.get_shape().as_list())
            else:
                shape = [1] + self.initial_state.c.get_shape().as_list()[1:]
                self.lstm_state_numeric = LSTMStateTuple(
                    np.zeros(shape, dtype='float32'),
                    np.zeros(shape, dtype='float32')
                )
                self.lstm_first_state_since_update = LSTMStateTuple(
                    np.zeros(shape, dtype='float32'),
                    np.zeros(shape, dtype='float32')
                )
        else:
            if self.model_name == ModelNames.A3C_CONV_LSTM:
                self.lstm_first_state_since_update.fill(0.)
                self.lstm_state_numeric.fill(0.)
            else:
                self.lstm_state_numeric[0].fill(0.)
                self.lstm_state_numeric[1].fill(0.)

                self.lstm_first_state_since_update[0].fill(0.)
                self.lstm_first_state_since_update[1].fill(0.)

    def reset(self):
        if self.recurrent:
            logger.debug("Resetting LSTM state")
            self.reset_lstm_state()

    def _add_trainable(self, layer, name=None):
        self.layers[layer.name] = layer
        if name:
            self.theta += tflearn.get_layer_variables_by_name(name)
        else:
            self.theta.append(layer.W)
            if layer.b:
                if isinstance(layer.b, list):
                    self.theta += layer.b
                else:
                    self.theta.append(layer.b)
        #logger.info('{}: {}'.format(layer.name, [t.name for t in self.theta]))


    def build_network(self, num_actions, input_shape):
        logger.debug('Input shape: {}'.format(input_shape))
        with tf.name_scope('ForwardInputs') as scope:
            self.forward_input_scope = scope
            self.inputs = tf.placeholder(tf.float32, [None] + input_shape, name="StateInput")
        logger.info('Building network: {}'.format(self.model_name))

        self.theta = []

        if self.model_name == ModelNames.NATURE:
            net = self._nature_model()
        elif self.model_name == ModelNames.A3C_FF:
            net = self._a3c_ff()
        elif self.model_name == ModelNames.A3C_LSTM:
            net = self._a3c_lstm()
        elif self.model_name == ModelNames.A3C_FF_SS:
            net = self._a3c_ff_ss()
        elif self.model_name == ModelNames.A3C_LSTM_SS:
            net = self._a3c_lstm_ss()
        elif self.model_name == ModelNames.A3C_CONV_LSTM:
            net = self._a3c_conv_lstm()
        elif self.model_name == ModelNames.A3C_CONV_LSTM_K:
            net = self._a3c_conv_lstm_k()
        elif self.model_name == ModelNames.A3C_SISWS:
            net = self._a3c_sisws()
        elif self.model_name == ModelNames.SMALL_FCN:
            net = self._small_fcn()
        elif self.model_name == ModelNames.A3C_SISWS_S:
            net = self._a3c_sisws_s()
        else:
            raise ValueError("Unknown model name {}".format(self.model_name))

        with tf.name_scope("Outputs"):
            with tf.name_scope("Policy"):
                if self.hp.safe_softmax:
                    self.pi = fc_layer(net, num_actions, activation='linear', name='pi_sa')
                    self._add_trainable(self.pi)
                    self.pi -= tf.stop_gradient(tf.expand_dims(tf.reduce_max(self.pi, reduction_indices=[1]), 1))
                    self.pi = tf.nn.softmax(self.pi)
                else:
                    self.pi = fc_layer(net, num_actions, activation='softmax', name='pi_sa')
                    self._add_trainable(self.pi)
            with tf.name_scope("Value"):
                if self.policy_weighted_val:
                    q_val = fc_layer(net, num_actions, activation='linear', name='q_sa')
                    self._add_trainable(q_val)
                    self.value = tf.reduce_sum(tf.mul(q_val, tf.stop_gradient(self.pi)),
                                               reduction_indices=1, name='v_s')
                else:
                    self.value = fc_layer(net, 1, activation='linear', name='v_s')
                    self._add_trainable(self.value)
                self.value = tflearn.reshape(self.value, [-1], 'FlattenedValue')

        if self.agent_name == 'GLOBAL':
            logger.info("Layer overview:")
            for key in self.layers.keys():
                logger.info('\t' + key)
        self.hidden_head = net

    def decoding_network(self, incoming, conv1, conv2):
        with tf.name_scope("DecodingNetwork"):
            # First we reshape the embedding into a volume with the shape of conv2
            net = tflearn.reshape(incoming, [-1] + conv2.get_shape().as_list()[1:], 'TransformationReshaped')
            if self.residual_prediction:
                net += conv2
            net = self.hp.activation(net, 'TransformationAct')

            # Then we perform a conv_2d_transpose (this is sometimes referred to as a DeConvolution layer)
            net = tflearn.conv_2d_transpose(net, 32, 4, strides=2, activation='linear',
                                            output_shape=conv1.get_shape().as_list()[1:],
                                            weight_decay=0., padding='valid', name='DeConv2')
            logger.warn("First deconv shape: {}".format(net.get_shape().as_list()))
            self._add_trainable(net)
            if self.residual_prediction:
                net += conv1
            net = self.hp.activation(net, name='DeConv2Act')

            # Then we do the latter again
            net = tflearn.conv_2d_transpose(net, 1, 8, strides=4, activation='linear',
                                            output_shape=[84, 84, 1], padding='valid', weight_decay=0., name='DeConv1')
            logger.warn("First deconv shape: {}".format(net.get_shape().as_list()))
            self._add_trainable(net)
            net = self.hp.activation(net, name='DeConv1Act')
        return net

    def build_frame_predictor(self):

        with tf.name_scope("EncodingHead"):
            # Embed those actions
            action_one_hot = tflearn.one_hot_encoding(self.actions, self.num_actions)
            action_embedding = tflearn.fully_connected(action_one_hot, 256, weight_decay=0.0, bias=False,
                                                       name='ActionEmbedding', activation='linear')
            self._add_trainable(action_embedding)

            # Embed the hidden layer head
            encoding = tflearn.fully_connected(self.hidden_head, 256, weight_decay=0.0, bias=False,
                                               name='EncodingEmbedding', activation='linear')
            self._add_trainable(encoding)

            # We will use the linear conv activation of conv1 and conv2 and inject those in their mirrored decoding layers
            conv1 = self.layers["{}/HiddenLayers/Conv1/BiasAdd:0".format(self.agent_name)]
            conv2 = self.layers["{}/HiddenLayers/Conv2/BiasAdd:0".format(self.agent_name)]

            # Now we can compute the 'transformation layer' which we will be put into the decoding stream
            transformation = tflearn.fully_connected(tf.mul(action_embedding, encoding),
                                                     np.prod(conv2.get_shape().as_list()[1:]), weight_decay=0.0,
                                                     activation='linear', name='Transformation')
            self._add_trainable(transformation)

        decoded_prediction = self.decoding_network(transformation, conv1, conv2)

        with tf.name_scope("FramePrediction"):
            self.predicted_frame = tf.transpose(decoded_prediction, [0, 3, 1, 2]) \
                                   + self.inputs[:, -1:, :, :]
            self.frame_target = tf.placeholder(tf.float32, [None, 1] + self.input_shape[1:])

        with tf.name_scope(self.loss_scope):
            frame_prediction_loss = tf.nn.l2_loss(self.frame_target[:, :, 2:-2, 2:-2] - self.predicted_frame[:, :, 2:-2, 2:-2],
                                                  name='FramePredictionLoss')
            self.summaries.append(tf.summary.scalar('{}/FramePredictionLoss'.format(self.agent_name),
                                                    frame_prediction_loss))

        self.fp_loss_coeff = tf.placeholder(tf.float32)
        self.loss += self.fp_loss_coeff * frame_prediction_loss

    def get_frame_prediction(self, state, action, session, rnn_state):
        feed_dict = {self.inputs: [state], self.actions: [action]}
        if self.recurrent:
            feed_dict[self.initial_state] = rnn_state
            feed_dict[self.n_steps] = [1]
        predicted_frame = session.run(self.predicted_frame, feed_dict=feed_dict)
        return predicted_frame

    def build_param_sync(self):
        with tf.name_scope("ParamSynchronization"):
            self.param_sync = tf.group(*[tf.assign(local_theta, global_theta, use_locking=False)
                                         for local_theta, global_theta in zip(self.theta, self.global_network.theta)])

    def build_param_update(self):
        with tf.name_scope("ParamUpdate"):
            self.minimize = self.optimizer.build_update_from_vars(self.theta, self.loss)

    def build_loss(self):
        """
        This function defines the computation graph that is needed to compute the loss of the approximators.
        """

        with tf.name_scope('BackwardInputs'):
            # The actions attribute is an array of length n
            self.actions = tf.placeholder(tf.int32, [None], name='Actions')
            # Rewards
            self.advantage_no_grad = tf.placeholder(tf.float32, [None], name="AdvantageNoGrad")
            self.n_step_returns = tf.placeholder(tf.float32, [None], name='NStepReturns')

        if self.clip_advantage:
            # I empirically found that it might help to clip the advantage that is used for the policy loss. This might
            # improve stability and consistency of the gradients
            logger.info("Clipping advantage in graph")
            self.advantage_no_grad = tf.clip_by_value(self.advantage_no_grad, -1., 1., name="ClippedAdvantage")

        with tf.name_scope("PolicyLoss"):
            # action matrix is n x a where each row corresponds to a time step and each column to an action
            action_mask = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name="ActionMask")
            # self.pi and log_pi are n x a matrices
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0), name="LogPi")
            # The entropy is added to encourage exploration
            entropy = -tf.reduce_sum(log_pi * self.pi, reduction_indices=1, name="Entropy")
            # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
            pi_loss = -(tf.reduce_sum(tf.mul(action_mask, log_pi), reduction_indices=1) * self.advantage_no_grad
                        + self.beta * entropy)

        with tf.name_scope("ValueLoss"):
            # A3C originally uses a factor 0.5 for the value loss. The l2_loss() method already does this
            advantage = self.n_step_returns - self.value
            value_loss = tf.square(advantage)
            if self.optimality_tightening:
                self.upper_limits = tf.placeholder(tf.float32, [None], name='UpperLimits')
                self.lower_limits = tf.placeholder(tf.float32, [None], name='LowerLimits')
                value_loss += self.hp.otc * (tf.nn.relu(self.lower_limits - self.value) ** 2 +
                                             tf.nn.relu(self.value - self.upper_limits) ** 2)
                value_loss /= (self.hp.otc + 1)

        # We can combine the policy loss and the value loss in a single expression
        with tf.name_scope("CombinedLoss"):

            # Add losses and
            self.loss = tf.reduce_sum(pi_loss + 0.5 * value_loss)

            # Add TensorBoard summaries
            self.summaries.append(tf.summary.scalar('{}/Loss'.format(self.agent_name), self.loss))
            self.summaries.append(tf.summary.scalar('{}/MaxAbsValue'.format(self.agent_name),
                                                    tf.reduce_max(tf.abs(self.value))))

    def get_action(self, state, session):
        """
        This function returns a single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's policy
        """
        if self.recurrent:

            pi, self.lstm_state_numeric = session.run(
                [
                    self.pi, self.lstm_state_variable
                ],
                feed_dict={
                    self.inputs: [state],
                    self.initial_state: self.lstm_state_numeric,
                    self.n_steps: [1]
                }
            )
            #print(self.lstm_state_numeric[0])
        else:
            pi = session.run(self.pi, feed_dict={self.inputs: [state]})

        action = np.random.choice(self.num_actions, p=pi[0])
        return action

    def get_value(self, state, session):
        """
        This function returns a single value that corresponds to the critic's valuation of the given state.
        :param state: The input state
        :return: State's value
        """
        if self.recurrent:
            # If we use a recurrent model
            v = session.run(self.value, feed_dict={self.inputs: [state], self.initial_state: self.lstm_state_numeric,
                                                      self.n_steps: [1]}
                            )[0]

            return v

        return session.run(self.value, feed_dict={self.inputs: [state]})[0]#[0]

    def get_value_and_action(self, state, session):
        """
        Returns the action and the value
        """
        if self.recurrent:
            value, pi, self.lstm_state_numeric = session.run(
                [
                    self.value, self.pi, self.lstm_state_variable
                ],
                feed_dict={
                    self.inputs: [state],
                    self.initial_state: self.lstm_state_numeric,
                    self.n_steps: [1]
                }
            )
        else:
            value, pi = session.run(
                [self.value, self.pi],
                feed_dict={self.inputs: [state]})

        action = np.random.choice(self.num_actions, p=pi[0])
        return value[0], action

    def get_value_and_action_and_visualize(self, state, session):
        """
        Returns the action and the value
        """
        activation_str = 'Relu' if self.hp.activation == tf.nn.relu else 'Elu'

        conv1 = session.graph.get_tensor_by_name(self.agent_name + "/HiddenLayers/{}:0".format(activation_str))
        conv2 = session.graph.get_tensor_by_name(self.agent_name + "/HiddenLayers/{}_1:0".format(activation_str))
        fc1 = session.graph.get_tensor_by_name(self.agent_name + "/HiddenLayers/FC3/{}:0".format(activation_str))
        lstm_out = None
        if self.recurrent:
            lstm = session.graph.get_tensor_by_name(self.agent_name + "/HiddenLayers/ReshapedLSTMOutput/Reshape:0")
            value, pi, self.lstm_state_numeric, conv1_out, conv2_out, fc1_out, lstm_out = session.run(
                [
                    self.value, self.pi, self.lstm_state_variable, conv1, conv2, fc1, lstm
                ],
                feed_dict={
                    self.inputs: [state],
                    self.initial_state: self.lstm_state_numeric,
                    self.n_steps: [1]
                }
            )
        else:
            value, pi, conv1_out, conv2_out, fc1 = session.run([self.value, self.pi, conv1, conv2, fc1],
                                                          feed_dict={self.inputs: [state]})

        action = np.random.choice(self.num_actions, p=pi[0])
        return value[0], action, conv1_out, conv2_out, pi[0], fc1_out, lstm_out

    def get_embedding(self, state, session):
        assert self.embedding_layer is not None, "No embedding layer was configured for TensorBoard embeddings"
        if self.recurrent:
            return session.run(self.embedding_layer, feed_dict={self.inputs: [state],
                                                                self.initial_state: self.lstm_state_numeric,
                                                                self.n_steps: [1]})
        return session.run(self.embedding_layer, feed_dict={self.inputs: [state]})

    def update_params(self, actions, states, lr, last_state, session, n_step_returns, values,
                      upper_limits=None, lower_limits=None, include_summaries=False):
        """
        Updates the parameters of the global network
        :param actions:             array of actions
        :param states:              array of states
        :param lr:                  actual learning rate
        """
        n_steps = len(actions)
        summaries = None

        if self.recurrent:

            fdict = {
                self.actions: actions,
                self.inputs: states,
                self.initial_state: self.lstm_first_state_since_update,
                self.n_steps: [n_steps],
                self.optimizer.learning_rate: lr,
                self.n_step_returns: n_step_returns,
                self.advantage_no_grad: n_step_returns - values
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: [s[-1:, :, :] for s in states[1:]] + [last_state[-1:, :, :]],
                              self.fp_loss_coeff: self.hp.fplc})
            if upper_limits and lower_limits:
                fdict.update({self.upper_limits: upper_limits, self.lower_limits: lower_limits})

            # Now we update our parameters AND we take the lstm_state
            if include_summaries:
                _, self.lstm_state_numeric, summaries = session.run(
                    [self.minimize, self.lstm_state_variable, self.merged_summaries], feed_dict=fdict)
            else:
                _, self.lstm_state_numeric = session.run([self.minimize, self.lstm_state_variable], feed_dict=fdict)

            # We also need to remember the LSTM state for the next backward pass
            self.lstm_first_state_since_update = deepcopy(self.lstm_state_numeric)
        else:
            fdict = {
                self.actions: actions,
                self.inputs: states,
                self.optimizer.learning_rate: lr,
                self.n_step_returns: n_step_returns,
                self.advantage_no_grad: n_step_returns - values
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: [s[-1:, :, :] for s in states[1:]] + [last_state[-1:, :, :]],
                              self.fp_loss_coeff: self.hp.fplc})
            if upper_limits and lower_limits:
                fdict.update({self.upper_limits: upper_limits, self.lower_limits: lower_limits})

            # Update the parameters
            if include_summaries:
                _, summaries = session.run([self.minimize, self.merged_summaries], feed_dict=fdict)
            else:
                session.run(self.minimize, feed_dict=fdict)


        return summaries

