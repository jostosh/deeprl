import tensorflow as tf
import tflearn
import numpy as np
from deeprl.common.logger import logger
from deeprl.approximators.layers import lstm, spatialsoftmax, custom_lstm
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
from copy import deepcopy
from deeprl.common.tensorflowutils import sequence_mask


class ModelNames:
    A3C_FF      = 'a3c_ff'
    NATURE      = 'nature'
    SMALL_FCN   = 'small_fcn'
    A3C_LSTM    = 'a3c_lstm'
    A3C_FF_SS   = 'a3c_ff_ss'
    A3C_LSTM_SS = 'a3c_lstm_ss'


class ActorCriticNN(object):

    def __init__(self, num_actions, agent_name, optimizer, hyper_parameters, global_network=None, distributed=False):
        self.num_actions = num_actions
        self.optimizer = optimizer
        self.global_network = global_network
        #self.session = session
        self.beta = hyper_parameters.beta
        self.summaries = []
        self.agent_name = agent_name
        self.model_name = hyper_parameters.model
        self.clip_advantage = hyper_parameters.clip_advantage
        self.recurrent = self.model_name in [ModelNames.A3C_LSTM, ModelNames.A3C_LSTM_SS]
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
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='linear', name='Conv1', weight_decay=0.0,
                                  bias_init=tf.constant_initializer(0.1), padding='valid')
            self._add_trainable(net)
            net = tf.nn.relu(net, name='Conv1Relu')

            # Add second convolutional layer
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='linear', name='Conv2', weight_decay=0.0,
                                  bias_init=tf.constant_initializer(0.1), padding='valid')
            self._add_trainable(net)
            net = tf.nn.relu(net, name='Conv2Relu')

            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 256, activation='relu', name='FC3', weight_decay=0.0,
                                          bias_init=tf.constant_initializer(0.1))
            self._add_trainable(net)
            self.embedding_layer = net

        return net, scope

    def _nature_model(self):
        """
        This is the more complicated model as taken from the nature paper in 2015.
        :param network_name:   Name of the network
        :return:               The feedforward model (last hidden layer as a graph node)
        """
        with tf.name_scope('ForwardInputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])

        with tf.name_scope('HiddenLayers'):
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name='Conv1')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu', name='Conv2')
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 3, strides=1, activation='relu', name='Conv3')
            self._add_trainable(net)
            net = tflearn.flatten(net)
            net = tflearn.fully_connected(net, 512, activation='relu', name='FC4')
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
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='linear', name='Conv1', weight_decay=0.0,
                                  bias_init=tf.constant_initializer(0.1), padding='valid')
            self._add_trainable(net)
            net = tf.nn.relu(net, 'Conv1Relu')
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='linear', name='Conv2', weight_decay=0.0,
                                  bias_init=tf.constant_initializer(0.1), padding='valid')
            self._add_trainable(net)
            net = spatialsoftmax(net)
            net = tflearn.fully_connected(net, 256, activation='relu', name='FC3', weight_decay=0.0,
                                          bias_init=tf.constant_initializer(0.1))
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
            self.initial_state_c = tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c")
            self.initial_state_h = tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            self.initial_state = LSTMStateTuple(
                self.initial_state_c,
                self.initial_state_h
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

            #logger.info(tflearn.get_layer_variables_by_name('LSTM4_{}'.format(self.agent_name)))

        #self.reset_lstm_state()
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
            self.initial_state_c = tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c")
            self.initial_state_h = tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            self.initial_state = LSTMStateTuple(
                self.initial_state_c,
                self.initial_state_h
            )
        with tf.name_scope('SequenceMasking'):
            self.n_steps = tf.placeholder(tf.int32, shape=[])
            seq_mask = tf.reshape(sequence_mask([self.n_steps], maxlen=self.t_max, dtype=tf.float32),
                                  [1, self.t_max, 1], name='SequenceMask')
        with tf.name_scope('ForwardInputs'):
            net = tf.transpose(self.inputs, [0, 2, 3, 1])
        with tf.name_scope('HiddenLayers'):
            net = tflearn.conv_2d(net, 32, 8, strides=4, activation='relu', name='Conv1', padding='valid', weight_decay=0.)
            self._add_trainable(net)
            net = tflearn.conv_2d(net, 64, 4, strides=2, activation='linear', name='Conv2', padding='valid', weight_decay=0.)
            self._add_trainable(net)
            #net = tflearn.flatten(net)
            net = spatialsoftmax(net)
            net = tflearn.fully_connected(net, 256, activation='relu', name='FC3')
            self._add_trainable(net)
            net = tf.mul(tflearn.reshape(net, [1, self.t_max, 256]), seq_mask, name='MaskedSequence')  # tf.expand_dims(net, 1)
            net, state = lstm(net, 256, initial_state=self.initial_state, return_state=True,
                              name='LSTM4', dynamic=True, return_seq=True)
            #logger.info(net._op.__dict__)
            self._add_trainable(net)
            #net = tflearn.reshape(net, [-1, 256])
            self.lstm_state_variable = state
            self.embedding_layer = net

        #self.reset_lstm_state()
        return net

    def reset_lstm_state(self):
        self.lstm_state_numeric = LSTMStateTuple(
            np.zeros([1, 256], dtype='float32'),
            np.zeros([1, 256], dtype='float32')
        )
        self.lstm_first_state_since_update = LSTMStateTuple(
            np.zeros([1, 256], dtype='float32'),
            np.zeros([1, 256], dtype='float32')
        )

    def reset(self):
        if self.recurrent:
            logger.debug("Resetting LSTM state")
            self.reset_lstm_state()

    def _add_trainable(self, layer, name=None):
        self.layers[layer.name] = layer
        if name:
            self.theta += tflearn.get_layer_variables_by_name(name)
        else:
            self.theta += [layer.W] + ([layer.b] if layer.b else [])
        #logger.info('{}: {}'.format(layer.name, [t.name for t in self.theta]))

    def _small_fcn(self):
        """
        This network works with the CartPole-v0/v1 environments (sort of)
        :param network_name:    The name of the network
        :return:                The network as a graph node
        """
        with tf.name_scope('HiddenLayers'):
            net = tflearn.fully_connected(self.inputs, 256, activation='tanh', name='FC1')
            self._add_trainable(net)
        self.embedding_layer = net
        return net

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
        else:
            net = self._small_fcn()

        with tf.name_scope("Outputs"):
            with tf.name_scope("Policy"):
                self.pi = tflearn.fully_connected(net, num_actions, activation='softmax', name='pi_sa', weight_decay=0.)
                self._add_trainable(self.pi)
            with tf.name_scope("Value"):
                if self.policy_weighted_val:
                    q_val = tflearn.fully_connected(net, num_actions, activation='linear', weight_decay=0.)
                    self._add_trainable(q_val)
                    self.value = tf.reshape(tf.reduce_sum(tf.mul(q_val, tf.stop_gradient(self.pi)),
                                                          reduction_indices=1), (-1, 1), name='v_s')
                else:
                    self.value = tflearn.fully_connected(net, 1, activation='linear', name='v_s')
                    self._add_trainable(self.value)

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
            net = tf.nn.relu(net, 'TransformationRelu')

            # Then we perform a conv_2d_transpose (this is sometimes referred to as a DeConvolution layer)
            net = tflearn.conv_2d_transpose(net, 32, 4, strides=2, activation='linear',
                                            output_shape=conv1.get_shape().as_list()[1:],
                                            weight_decay=0., padding='valid', name='DeConv2')
            self._add_trainable(net)
            if self.residual_prediction:
                net += conv1
            net = tf.nn.relu(net, name='DeConv2Relu')

            # Then we do the latter again
            net = tflearn.conv_2d_transpose(net, 1, 8, strides=4, activation='linear',
                                            output_shape=[84, 84, 1], padding='valid', weight_decay=0., name='DeConv1')
            self._add_trainable(net)
            net = tf.nn.relu(net, name='DeConv1Relu')
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
            frame_prediction_loss = tf.reduce_sum((self.frame_target - self.predicted_frame) ** 2,
                                                  name='FramePredictionLoss')
            self.summaries.append(tf.summary.scalar('{}/FramePredictionLoss'.format(self.agent_name),
                                                    frame_prediction_loss))

        self.loss += frame_prediction_loss

    def build_param_sync(self):
        with tf.name_scope("ParamSynchronization"):
            self.param_sync = [tf.assign(local_theta, global_theta, use_locking=False)
                               for local_theta, global_theta in zip(self.theta, self.global_network.theta)]

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
            self.rewards = tf.placeholder(tf.float32, [None], name='Rewards')
            self.initial_return = tf.placeholder(tf.float32, name='InitialReturn')

        with tf.name_scope("Advantage_n-step"):
            def reward_fn(n_step_t, reward):
                return n_step_t * self.gamma + reward

            rewards = tf.clip_by_value(self.rewards, -.1, 1.) if self.clip_rewards else self.rewards
            n_step_returns = tf.scan(reward_fn, tf.reverse_v2(rewards, [0]),
                                     initializer=self.initial_return)
            advantage = tf.reshape(tf.reverse_v2(n_step_returns, [0]), [-1, 1]) - self.value


        # The advantage is simply the estimated value minus the bootstrapped returns
        # advantage = self.n_step_returns - self.value
        # advantage_no_grad = self.n_step_returns - tf.stop_gradient(self.value)

        '''
        if self.clip_advantage:
            # I empirically found that it might help to clip the advantage that is used for the policy loss. This might
            # improve stability and consistency of the gradients
            logger.info("Clipping advantage in graph")
            self.advantage_no_grad = tf.clip_by_value(self.advantage_no_grad, -1., 1., name="ClippedAdvantage")
        '''

        with tf.name_scope("PolicyLoss"):
            # action matrix is n x a where each row corresponds to a time step and each column to an action
            action_mask = tf.one_hot(self.actions, self.num_actions, 1.0, 0.0, name="ActionMask")
            # self.pi and log_pi are n x a matrices
            log_pi = tf.log(tf.clip_by_value(self.pi, 1e-20, 1.0), name="LogPi")
            # The entropy is added to encourage exploration
            entropy = -tf.reshape(tf.reduce_sum(log_pi * self.pi, reduction_indices=1), (-1, 1), name="Entropy")
            # Define the loss for the policy (minus is needed to perform *negative* gradient descent == gradient ascent)
            pi_loss = tf.neg(
                tflearn.reshape(tf.reduce_sum(action_mask * log_pi, reduction_indices=1), (-1, 1))
                * tf.stop_gradient(advantage) #self.advantage_no_grad
                + self.beta * entropy, name='PiLoss')

        with tf.name_scope("ValueLoss"):
            value_loss = tf.square(advantage)
            if self.optimality_tightening:
                self.upper_limits = tf.placeholder(tf.float32, [None], name='UpperLimits')
                self.lower_limits = tf.placeholder(tf.float32, [None], name='LowerLimits')
                value_loss += 4 * (tf.nn.relu(tf.reshape(self.lower_limits, [-1, 1]) - self.value) ** 2 +
                                   tf.nn.relu(self.value - tf.reshape(self.upper_limits, [-1, 1])) ** 2)
                value_loss /= 5

        # We can combine the policy loss and the value loss in a single expression
        with tf.name_scope("CombinedLoss"):

            # Add losses and use a factor 0.5 for the value loss as suggested by Mnih
            self.loss = tf.reduce_sum(pi_loss + 0.5 * value_loss, name='Loss')

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
            return session.run(self.value,
                                    feed_dict={
                                        self.inputs: [state],
                                        self.initial_state: self.lstm_state_numeric,
                                        self.n_steps: [1]
                                    })[0][0]

        return session.run(self.value, feed_dict={self.inputs: [state]})[0][0]

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
        return value[0][0], action

    def get_embedding(self, state, session):
        assert self.embedding_layer is not None, "No embedding layer was configured for TensorBoard embeddings"
        if self.recurrent:
            return session.run(self.embedding_layer, feed_dict={self.inputs: [state],
                                                                self.initial_state: self.lstm_state_numeric,
                                                                self.n_steps: [1]})
        return session.run(self.embedding_layer, feed_dict={self.inputs: [state]})

    def update_params(self, actions, states, lr, last_state, session, rewards, initial_return,
                      upper_limits=None, lower_limits=None):
        """
        Updates the parameters of the global network
        :param actions:             array of actions
        :param states:              array of states
        :param lr:                  actual learning rate
        """
        n_steps = len(actions)

        if self.recurrent:

            fdict = {
                self.actions: actions,
                self.inputs: states,
                self.initial_state: self.lstm_first_state_since_update,
                self.n_steps: [n_steps],
                self.optimizer.learning_rate: lr,
                self.initial_return: initial_return,
                self.rewards: rewards
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: [s[-1:, :, :] for s in states[1:]] + [last_state[-1:, :, :]]})
            if upper_limits and lower_limits:
                fdict.update({self.upper_limits: upper_limits, self.lower_limits: lower_limits})

            # Now we update our parameters AND we take the lstm_state
            _, self.lstm_state_numeric = session.run([self.minimize, self.lstm_state_variable], feed_dict=fdict)

            # We also need to remember the LSTM state for the next backward pass
            self.lstm_first_state_since_update = deepcopy(self.lstm_state_numeric)
        else:
            fdict = {
                self.actions: actions,
                self.inputs: states,
                self.optimizer.learning_rate: lr,
                self.initial_return: initial_return,
                self.rewards: rewards
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: [s[-1:, :, :] for s in states[1:]] + [last_state[-1:, :, :]]})
            if upper_limits and lower_limits:
                fdict.update({self.upper_limits: upper_limits, self.lower_limits: lower_limits})

            # Update the parameters
            session.run([self.minimize], feed_dict=fdict)

        return None #summaries
        #writer.add_summary(summaries, t)

    def compute_delta(self, n_step_return, actions, states, values, learning_rate_var, lr, last_state, session):
        """
        Updates the parameters of the global network
        :param n_step_return:       n-step returns
        :param actions:             array of actions
        :param states:              array of states
        :param values:              array of values
        :param learning_rate_var:   learning rate placeholder,
        :param lr:                  actual learning rate
        """
        n_steps = len(n_step_return)

        if self.recurrent:
            # First we need to pad the sequences we got
            n_pad = self.t_max - n_steps
            pad1d = np.zeros(n_pad)
            n_step_return   = np.concatenate([n_step_return, pad1d]).reshape((self.t_max, 1))
            actions         = np.concatenate([actions, pad1d])
            values          = np.concatenate([values, pad1d]).reshape((self.t_max, 1))
            states          = np.concatenate([states, np.zeros((self.t_max - n_steps,) + states[0].shape)])

            fdict = {
                self.n_step_returns: n_step_return.reshape((self.t_max, 1)),
                self.actions: actions,
                self.inputs: states,
                self.advantage_no_grad: (n_step_return - values).reshape((self.t_max, 1)),
                self.initial_state: self.lstm_first_state_since_update,
                self.n_steps: [n_steps]
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: [s[-1, :, :] for s in states[1:] + [last_state]]})

            # Now we update our parameters AND we take the lstm_state
            gradients, summaries, self.lstm_state_numeric = session.run([
                self.local_gradients,
                self.merged_summaries,
                self.lstm_state_variable
            ],
                feed_dict=fdict
            )
            # We also need to remember the LSTM state for the next backward pass
            self.lstm_first_state_since_update = deepcopy(self.lstm_state_numeric)
        else:
            fdict = {
                self.n_step_returns: n_step_return.reshape((n_steps, 1)),
                self.actions: actions,
                self.inputs: states,
                self.advantage_no_grad: (n_step_return - values).reshape((n_steps, 1))
            }
            if self.frame_prediction:
                fdict.update({self.frame_target: states[1:] + [last_state]})

            # Update the parameters
            gradients, summaries = session.run([
                self.local_gradients,
                self.merged_summaries
            ],
                feed_dict=fdict
            )


        return summaries, gradients
        #writer.add_summary(summaries, t)
