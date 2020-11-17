import tensorflow as tf
import os
import numpy


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

    def model_parameters(self, lstm_out_is_external):

        self.lstm_out_is_external = lstm_out_is_external


    def transition_model(self):
        transition_model_input = tf.keras.layers.Input(shape=(None, self.image_width, self.image_width, 1), name='transition_model_input')
        action_in = tf.keras.layers.Input(shape=(None, 1), name='action_in')
        lstm_hidden_state_h_in = tf.keras.layers.Input(shape=(150), name='lstm_hidden_state_h_in')
        lstm_hidden_state_c_in = tf.keras.layers.Input(shape=(150), name='lstm_hidden_state_c_in')
        lstm_hidden_state_h_out = tf.keras.layers.Input(shape=(150), name='lstm_hidden_state_h_out')
        lstm_hidden_state_c_out = tf.keras.layers.Input(shape=(150), name='lstm_hidden_state_c_out')
        condition_lstm = tf.keras.layers.Input(shape=(1), name='condition_lstm')
       # condition_lstm =condition_lstm.numpy()
        #condition_lstm_out = tf.keras.layers.Dense(1)(condition_lstm)

        lstm_hidden_state_in = [lstm_hidden_state_h_in, lstm_hidden_state_c_in]

        # Convolutional encoder
        conv1_layer = tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same', name='conv1')
        conv2_layer = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')
        conv3_layer = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')
        x = tf.keras.layers.TimeDistributed(conv1_layer)(transition_model_input)
        x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x)
        x = tf.keras.layers.TimeDistributed(conv2_layer)(x)
        x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
        conv3 = tf.keras.layers.TimeDistributed(conv3_layer)(x)
        latent_space = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(conv3)

        latent_space_shape = latent_space.get_shape()

        fc_1 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_1'))(action_in)

        fc_2 = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_2'))(latent_space)

        concat_0 = tf.concat([fc_1, fc_2], axis=2, name='concat_0')

        # LSTM (option 1)
        my_LSTMCell = tf.keras.layers.LSTMCell(self.lstm_hidden_state_size)
        my_RNN_layer = tf.keras.layers.RNN(my_LSTMCell, return_sequences=False, return_state=True, name='rnn_layer')
        _, h_state, c_state = my_RNN_layer(inputs=concat_0, initial_state=lstm_hidden_state_in)

        lstm_out_internal = h_state
        lstm_out_external = lstm_hidden_state_h_out

        #final_memory_state = tf.cond(condition_lstm[2] == 1, lambda: lstm_out_external,
                                    # lambda: lstm_out_internal)

        final_memory_state = tf.keras.backend.switch(condition=tf.keras.backend.equal(condition_lstm[0], 7), then_expression=lambda: lstm_out_external, else_expression=lambda: lstm_out_internal)

        #final_memory_state = h_state
        concat2_part1 = final_memory_state[:, -150:]
        concat2_part2 = latent_space[:, -1, :]
        concat_2 = tf.concat([concat2_part1, concat2_part2], axis=1)

        # State representation
        state_representation = tf.keras.layers.Dense(1000, activation="tanh", name='fc_3')(concat_2)

        fc_4 = tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_4')(state_representation)
        fc_4 = tf.reshape(fc_4, [-1, 8, 8, 4])  # go to shape of the latent space

        # Convolutional decoder
        dec_input = fc_4

        x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv1')(x)
        x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv2')(x)
        transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same',
                                                                  activation='sigmoid', name='deconv3')(x)

        model_transition = tf.keras.Model(
            inputs=[transition_model_input, action_in, lstm_hidden_state_h_in, lstm_hidden_state_c_in,
                    lstm_hidden_state_h_out, lstm_hidden_state_c_out, condition_lstm],
            outputs=[h_state, c_state, state_representation, transition_model_output, condition_lstm[0]],

            name="model_transition")


        return model_transition

    def policy_model(self):

        # Inputs
        state_representation_input = tf.keras.layers.Input(shape=(1000), batch_size=None, name='state_representation_input')
        self.policy_input = tf.keras.layers.LayerNormalization(name='norm_policy')(state_representation_input)

        # Fully connected layers
        fc_5 = tf.keras.layers.Dense(1000, activation="relu", name='fc_5')(self.policy_input)
        fc_6 = tf.keras.layers.Dense(1000, activation="relu", name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(self.dim_a, activation="tanh", name='fc_7')(fc_6)
        self.policy_output = fc_7

        # Model creation
        model_policy = tf.keras.Model(inputs=[state_representation_input], outputs=[self.policy_output])
        return model_policy

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