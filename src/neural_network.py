import tensorflow as tf
import os


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

        # Build Neural Network
        self._build_transition_model()
        self._build_policy()

        # Initialize optimizers
        self.policy_optimizer = tf.keras.optimizers.SGD(learning_rate=policy_learning_rate)
        self.transition_model_optimizer = tf.keras.optimizers.Adam(learning_rate=transition_model_learning_rate)  # TODO learning rate from config

    def _build_transition_model(self):
        transition_model_input = tf.keras.layers.Input(shape=(None, self.image_width, self.image_width, 1), name='transition_model_input')
        action_in = tf.keras.layers.Input(shape=(None, 1), name='action_in')
        lstm_hidden_state_in = [tf.keras.layers.Input(shape=self.lstm_hidden_state_size, name='lstm_hidden_state_h_in'),
                                tf.keras.layers.Input(shape=self.lstm_hidden_state_size, name='lstm_hidden_state_c_in')]

        # Convolutional encoder
        conv1_layer = tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same', name='conv1')
        conv2_layer = tf.keras.layers.Conv2D(8, (3, 3), strides=2, padding='same', name='conv2')
        conv3_layer = tf.keras.layers.Conv2D(4, (3, 3), strides=2, padding='same', activation='sigmoid', name='conv3')

        x = tf.keras.layers.TimeDistributed(conv1_layer)(transition_model_input)
        x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x)
        x = tf.keras.layers.TimeDistributed(conv2_layer)(x)
        x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
        conv3 = tf.keras.layers.TimeDistributed(conv3_layer)(x)
        latent_space = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(conv3)

        latent_space_shape = latent_space.get_shape()

        fc_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(latent_space_shape[-1], activation='tanh', name='fc_1'))(action_in)

        fc_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(latent_space_shape[-1], activation='tanh', name='fc_2'))(latent_space)

        concat_0 = tf.concat([fc_1, fc_2], axis=2, name='concat_0')

        # LSTM
        cell = tf.keras.layers.LSTMCell(self.lstm_hidden_state_size)
        lstm = tf.keras.layers.RNN(cell, return_sequences=False, return_state=True, name='rnn_layer')

        # Init RNN layer
        _, lstm_hidden_state_h_out, lstm_hidden_state_c_out = lstm(inputs=concat_0, initial_state=lstm_hidden_state_in)

        # Prepare state representation input
        concat2_part1 = lstm_hidden_state_h_out[:, -self.lstm_hidden_state_size:]
        concat2_part2 = latent_space[:, -1, :]
        concat_2 = tf.concat([concat2_part1, concat2_part2], axis=1)

        # State representation
        state_representation = tf.keras.layers.Dense(1000, activation="tanh", name='fc_3')(concat_2)

        fc_4 = tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_4')(state_representation)
        fc_4 = tf.reshape(fc_4, [-1, conv3.shape[2], conv3.shape[3], conv3.shape[4]])  # go to shape of the latent space

        # Convolutional decoder
        dec_input = fc_4

        x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv1')(x)
        x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv2')(x)
        transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same',
                                                                  activation='sigmoid', name='deconv3')(x)
        # Create model
        self.NN_transition_model = tf.keras.Model(
            inputs=[transition_model_input, action_in, lstm_hidden_state_in],
            outputs=[[lstm_hidden_state_h_out, lstm_hidden_state_c_out], state_representation, transition_model_output],
            name='model_transition')

    def _build_policy(self):
        # Inputs
        state_representation_input = tf.keras.layers.Input(shape=1000, batch_size=None, name='state_representation_input')
        policy_input = tf.keras.layers.LayerNormalization(name='norm_policy')(state_representation_input)

        # Fully connected layers
        fc_5 = tf.keras.layers.Dense(1000, activation='relu', name='fc_5')(policy_input)
        fc_6 = tf.keras.layers.Dense(1000, activation='relu', name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(self.dim_a, activation='tanh', name='fc_7')(fc_6)
        policy_output = fc_7

        # Create model
        self.NN_policy = tf.keras.Model(inputs=[state_representation_input], outputs=[policy_output])

    def save_transition_model(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)

        self.saver_transition_model.save(self.sess, self.network_loc + '_transition_model')

    def _load_transition_model(self):
        self.saver_transition_model.restore(self.sess, self.network_loc + '_transition_model')

    def save_policy(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)

        self.saver_transition_model.save(self.sess, self.network_loc + '_policy')

    def _load_policy(self):
        self.saver_transition_model.restore(self.sess, self.network_loc + '_transition_policy')