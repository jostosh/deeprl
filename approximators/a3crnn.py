from deeprl.approximators.actorcritic import ActorCriticApproximator
from deeprl.approximators.layers.core import *
from deeprl.approximators.layers.convlstm import ConvLSTM2D
from copy import deepcopy
from deeprl.common.logger import logger
from tensorflow.python.ops.rnn_cell import LSTMStateTuple
import abc


class A3CRecurrent(ActorCriticApproximator):

    def __init__(self, session, num_actions, optimizer, global_approximator, name):
        super().__init__(session, num_actions, optimizer, global_approximator, name)
        self.rnn_state_numeric = None
        self.rnn_state_op = None
        self.rnn_state_init = None
        self.rnn_first_state_since_last_update = None
        self.n_steps = None

    def get_action(self, state):
        """
        This function returns a_t single array reflecting the stochastic policy pi for the given state.
        :param state: The input state
        :return: State's_t policy
        """
        pi, self.rnn_state_numeric = self.session.run(
            [self.pi, self.rnn_state_op],
            feed_dict={
                self.states: [state],
                self.rnn_state_init: self.rnn_state_numeric,
                self.n_steps: [1]
            }
        )
        action = np.random.choice(self.num_actions, p=pi[0])
        return action

    def get_value(self, state):
        """
        This function returns a_t single value that corresponds to the critic's_t valuation of the given state.
        :param state: The input state
        :return: State's_t value
        """
        return self.session.run(
            self.value,
            feed_dict={
                self.states: [state],
                self.rnn_state_init: self.rnn_state_numeric,
                self.n_steps: [1]
            }
        )[0]

    def get_value_and_action(self, state):
        """
        Returns the value and action given a_t state
        :param state: State input
        :return: Value and action as float and integer, respectively
        """
        value, pi, self.rnn_state_numeric = self.session.run(
            [
                self.value, self.pi, self.rnn_state_op
            ],
            feed_dict={
                self.states: [state],
                self.rnn_state_init: self.rnn_state_numeric,
                self.n_steps: [1]
            }
        )
        value, pi = self.session.run([self.value, self.pi], feed_dict={self.states: [state]})
        action = np.random.choice(self.num_actions, p=pi[0])
        return value[0], action

    def get_embedding(self, state):
        assert self.embedding_layer is not None, "No embedding layer was configured for TensorBoard embeddings"
        return self.session.run(
            self.embedding_layer,
            feed_dict={
                self.states: [state],
                self.rnn_state_init: self.rnn_state_numeric,
                self.n_steps: [1]
            }
        )

    def update_params(self, actions, states, values, n_step_returns, lr, include_summaries, **kwargs):
        """
        Updates the global parameters
        :param actions: Actions that were chosen
        :param states: States observed
        :param values: Values estimated
        :param lr: Learning rate
        :param n_step_returns: n-step returns
        :param include_summaries: Whether to include summaries in output for TensorBoard
        :param kwargs: Other arguments
        :return:
        """
        feed_dict = self._update_feed_dict(actions, states, values, n_step_returns, lr).update({
            self.n_steps: len(actions),
            self.rnn_state_init: self.rnn_first_state_since_last_update
        })
        # Update the parameters
        if include_summaries:
            _, summaries, self.rnn_first_state_since_last_update = self.session.run(
                [self.minimize, self.merged_summaries, self.rnn_state_op], feed_dict=feed_dict
            )
        else:
            _, self.rnn_first_state_since_last_update = self.session.run(
                [self.minimize, self.rnn_state_op], feed_dict=feed_dict
            )
        self.rnn_state_numeric = deepcopy(self.rnn_first_state_since_last_update)

    def _reset_rnn_state(self):
        """
        Resets the RNN state to zero
        """
        [s.fill(0.) for s in self.rnn_state_numeric]
        [s.fill(0.) for s in self.rnn_first_state_since_last_update]

    def _define_rnn_state(self):
        shape = [1] + self.rnn_state_init.c.get_shape().as_list()[1:]
        self.rnn_state_numeric = LSTMStateTuple(
            np.zeros(shape, dtype='float32'),
            np.zeros(shape, dtype='float32')
        )
        self.rnn_first_state_since_last_update = LSTMStateTuple(
            np.zeros(shape, dtype='float32'),
            np.zeros(shape, dtype='float32')
        )

    def reset(self):
        logger.debug("Resetting RNN state")
        self._reset_rnn_state()

    @abc.abstractmethod
    def _build_hidden_layers(self):
        """
        Builds hidden layers
        :return: Tensor Op of last layer
        """


class A3CLSTM(A3CRecurrent):

    def __init__(self, session):
        super().__init__(**kwargs)

    def _build_hidden_layers(self):

        with tf.name_scope('LSTMInput'):
            # An LSTM layers's_t 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.rnn_state_init = LSTMStateTuple(
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c"),
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            )
            self.n_steps = tf.placeholder(tf.int32, shape=[1])

        net, scope = self._nips_hidden_layers()

        with tf.name_scope(scope) as scope:
            net = tf.reshape(net, [1, -1, 256], "ReshapedLSTMInput")
            net, self.rnn_state_op = custom_lstm(
                net, 256, initial_state=self.rnn_state_init, name='LSTM4_{}'.format(self.name),
                sequence_length=self.n_steps
            )
            self._add_trainable(net)
            net = tf.reshape(net, [-1, 256], name="ReshapedLSTMOutput")
            self.embedding_layer = net

        return net


class A3CLSTMSS(A3CRecurrent):

    def __init__(self, session):
        super().__init__(**kwargs)

    def _build_hidden_layers(self):
        """
        This is the feedforward model taken from "Asynchronous Methods for Reinforcement Learning"
        :param network_name:    Name of the network
        :return:                The feedforward model (last hidden layer) as a_t graph node
        """
        with tf.name_scope('LSTMStateInput'):
            # An LSTM layers's_t 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.initial_state = LSTMStateTuple(
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_c"),
                tf.placeholder(tf.float32, shape=[1, 256], name="InitialLSTMState_h")
            )
            self.n_steps = tf.placeholder(tf.int32, shape=[1])
        with tf.name_scope('ForwardInputs'):
            net = tf.transpose(self.states, [0, 2, 3, 1])
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
                                     name='LSTM4_{}'.format(self.name),
                                     sequence_length=self.n_steps)
            self._add_trainable(net)
            net = tflearn.reshape(net, [-1, 256], name="ReshapedLSTMOutput")
            self.lstm_state_variable = state
            self.embedding_layer = net
        return net


class A3CConvLSTM(A3CRecurrent):

    def _build_hidden_layers(self):
        with tf.name_scope('Inputs'):
            net = tf.transpose(self.states, [0, 2, 3, 1])

        with tf.name_scope('LSTMInput'):
            # An LSTM layers's_t 'state' is defined by the activation of the cells 'c' (256) plus the output of the cell
            # 'h' (256), which are both influencing the layer in the forward/backward pass.
            self.n_steps = tf.placeholder(tf.int32, shape=[1])

        with tf.name_scope('HiddenLayers'):
            net = conv_layer(net, 32, 8, 4, activation=self.hp.activation, name='Conv1')
            self._add_trainable(net)
            net = tf.reshape(net, [-1, 1] + net.get_shape().as_list()[1:])

            self.initial_state = LSTMStateTuple(c=tf.placeholder(tf.float32, [None, 9, 9, 64]),
                                                h=tf.placeholder(tf.float32, [None, 9, 9, 64]))

            convlstm_layer = ConvLSTM2D(nb_filter=64, nb_row=4, nb_col=4, subsample=(2, 2), dim_ordering='tf',
                                        state_ph=self.initial_state, return_sequences=True, nb_row_i=3, nb_col_i=3,
                                        inner_activation=tf.nn.sigmoid)
            net = convlstm_layer(net)

            self.theta += convlstm_layer.trainable_weights
            self.lstm_state_variable = convlstm_layer.state_out

            net = tf.reshape(net, [-1, 9 * 9 * 64])
            net = fc_layer(net, 256, activation=self.hp.activation, name='FC3')
            self.embedding_layer = net

        return net


