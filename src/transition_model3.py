import numpy as np
from tools.functions import observation_to_gray, FastImagePlot
from buffer import Buffer
import cv2
import tensorflow as tf
import time

"""
Transition model.
"""


class TransitionModel:
    def __init__(self, training_sequence_length, lstm_hidden_state_size, crop_observation, image_width,
                 show_transition_model_output, show_observation, resize_observation, occlude_observation, dim_a,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode):

        self.lstm_h_size = lstm_hidden_state_size
        self.dim_a = dim_a
        self.training_sequence_length = training_sequence_length
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode

        # System model parameters
        #self.lstm_hidden_state = np.zeros([1, 2 * self.lstm_h_size])

        #self.lstm_hidden_state = [tf.cast(tf.reshape(tf.zeros(self.lstm_h_size), [1, self.lstm_h_size]), tf.float32), tf.cast(tf.reshape(tf.zeros(self.lstm_h_size), [1, self.lstm_h_size]), tf.float32)]


        self.image_width = image_width  # we assume that images are squares

        # High-dimensional observation initialization
        self.resize_observation = resize_observation
        self.show_observation = show_observation
        self.show_ae_output = show_transition_model_output
        self.t_counter = 0
        self.crop_observation = crop_observation
        self.occlude_observation = occlude_observation

        # Buffers
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 1]))
        self.transition_model_buffer_sampling_rate = buffer_sampling_rate
        self.transition_model_sampling_size = buffer_sampling_size

        if self.show_observation:
            self.state_plot = FastImagePlot(1, np.zeros([self.image_width, self.image_width]),
                                            self.image_width, 'Image State', vmax=1.0)

        if self.show_ae_output:
            self.ae_output_plot = FastImagePlot(3, np.zeros([self.image_width, self.image_width]),
                                                self.image_width, 'Autoencoder Output', vmax=1.0)

    def _preprocess_observation(self, observation):
        # print('TRANSITION-MODEL: _preprocess_observation')
        if self.occlude_observation:
            observation[48:, :, :] = np.zeros(
                [48, 96, 3]) + 127  # TODO: occlusion should be a function of the input size

        if self.crop_observation:
            observation = observation[:, 80:-80]  # TODO: these numbers should not be hard coded

        if self.resize_observation:
            observation = cv2.resize(observation, (self.image_width, self.image_width), interpolation=cv2.INTER_AREA)

        self.processed_observation = observation_to_gray(observation, self.image_width)
        self.last_states.add(self.processed_observation)
        self.network_input = np.array(self.last_states.buffer)

        self.network_input = tf.convert_to_tensor(self.network_input, dtype=tf.float32)

    def _refresh_image_plots(self, ae_model_output):
        # print('TRANSITION-MODEL: _refresh_image_plots')
        if self.t_counter % 4 == 0 and self.show_observation:
            self.state_plot.refresh(self.processed_observation)

        if (self.t_counter + 2) % 4 == 0 and self.show_ae_output:
            self.ae_output_plot.refresh(ae_model_output)

    def _train_model_from_database(self, neural_network, database, i_episode, t):
        # print('TRANSITION-MODEL: _train_model_from_database')
        episodes_num = len(database)

        print('Training Transition Model...')
        for i in range(self.number_training_iterations):  # Train
            if i % (self.number_training_iterations / 20) == 0:
                print('Progress Transition Model training: %i %%' % (i / self.number_training_iterations * 100))

                # print(self.lstm_hidden_state)
                if i == 0 and i_episode == 0:
                    bandera2 = 1
                else:
                    bandera2 = 0

                # if i == 249:
                # bandera3 = 1
                # else:
                # bandera3 = 0

            observations, actions, predictions = [], [], []

            # Sample batch from database
            for i in range(self.transition_model_sampling_size):
                count = 0
                while True:
                    count += 1
                    if count > 1000:  # check if it is possible to sample
                        print('Database too small for training!')
                        return

                    selected_episode = round(
                        np.random.uniform(-0.49, episodes_num - 1))  # select and episode from the database randomly
                    episode_trajectory_length = len(database[selected_episode])

                    if episode_trajectory_length > self.training_sequence_length + 2:
                        break

                sequence_start = round(
                    np.random.uniform(0, episode_trajectory_length - self.training_sequence_length - 1))

                sequence = database[selected_episode][
                           sequence_start:sequence_start + self.training_sequence_length + 1]  # get samples from database

                observation_seq = []
                action_seq = []

                # Separate observations, actions and expected observation predictions from sampled batch
                for i in range(len(sequence)):
                    observation_seq.append(sequence[i][0])
                    action_seq.append(sequence[i][1])

                observations.append(observation_seq[:-1])
                actions.append(action_seq[:-1])
                predictions.append(observation_seq[-1])

            observations = np.array(observations)  # TODO: these are required for LSTM training
            actions = np.array(actions)
            predictions = np.array(predictions)

            action_in = np.reshape(actions,
                                   [self.transition_model_sampling_size * self.training_sequence_length, self.dim_a])

            # action_in = np.reshape(actions,
            # [self.transition_model_sampling_size, self.training_sequence_length, self.dim_a])

            # Train transition model

            input_to_encoder = np.reshape(observations,
                                          [self.transition_model_sampling_size * self.training_sequence_length,
                                           self.image_width, self.image_width, 1])
            # input_to_encoder = np.reshape(observations,
            # [self.transition_model_sampling_size, self.training_sequence_length,
            # self.image_width, self.image_width, 1])
            input_to_encoder = tf.convert_to_tensor(input_to_encoder, dtype=tf.float32)
            input_to_encoder_shape = tuple(input_to_encoder.get_shape().as_list())

            action_in = tf.convert_to_tensor(action_in, dtype=tf.float32)

            neural_network.model_parameters(batch_size=self.transition_model_sampling_size,
                                            sequence_length=self.training_sequence_length)

            if bandera2 == 1:
                self.transition_model_training = neural_network.transition_model()

            # Print model summary, notice the shape of the input layer
            # self.transition_model_training.summary()
            transition_model_label = np.reshape(predictions, [self.transition_model_sampling_size, self.image_width,
                                                              self.image_width, 1]),

            transition_model_label = tf.convert_to_tensor(transition_model_label, dtype=tf.float32)
            # save model as a png
            tf.keras.utils.plot_model(self.transition_model_training, to_file='model_plot2.png', show_shapes=True,
                                      show_layer_names=True)

            # TRAIN transition model
            optimizer_transition_model = tf.keras.optimizers.Adam(learning_rate=0.0005)  # irenee 0.0005 --> 0.005

            with tf.GradientTape() as tape_transition:

                final_memory_state, _, prediction_value = self.transition_model_training([input_to_encoder, action_in])
                # print(lstm_hidden_state_out_check)

                current_loss = tf.reduce_mean(tf.square(prediction_value - transition_model_label))
                grads = tape_transition.gradient(current_loss, self.transition_model_training.trainable_variables)

            optimizer_transition_model.apply_gradients(zip(grads, self.transition_model_training.trainable_variables))
            # self.conv1_weights = self.transition_model_training.get_layer('conv1').get_weights()
            # print(whole_sequence)
            # print(final_memory_state.shape)
        # if bandera3 == 1:
        # self.training_weights = self.transition_model_training.get_weights()
        # print('PILLO WEIGHTSSS')
        self.conv1_weights = self.transition_model_training.get_layer('conv1').get_weights()
        self.norm_conv1_weights = self.transition_model_training.get_layer('norm_conv1').get_weights()
        self.conv2_weights = self.transition_model_training.get_layer('conv2').get_weights()
        self.norm_conv2_weights = self.transition_model_training.get_layer('norm_conv2').get_weights()
        self.conv3_weights = self.transition_model_training.get_layer('conv3').get_weights()

        self.fc_1_weights = self.transition_model_training.get_layer('fc_1').get_weights()
        self.fc_2_weights = self.transition_model_training.get_layer('fc_2').get_weights()
        self.rnn_layer_weights = self.transition_model_training.get_layer('rnn_layer').get_weights()

        self.fc_3_weights = self.transition_model_training.get_layer('fc_3').get_weights()
        self.fc_4_weights = self.transition_model_training.get_layer('fc_4').get_weights()
        self.deconv1_weights = self.transition_model_training.get_layer('deconv1').get_weights()
        self.norm_deconv1_weights = self.transition_model_training.get_layer('norm_deconv1').get_weights()
        self.deconv2_weights = self.transition_model_training.get_layer('deconv2').get_weights()
        self.norm_deconv2_weights = self.transition_model_training.get_layer('norm_deconv2').get_weights()
        self.deconv3_weights = self.transition_model_training.get_layer('deconv3').get_weights()
        # weights = self.transition_model_training.dense1.get_weights()

        # print('weight conv1 just after training')
        # print(self.conv1_weights[0][0][0][0][0])

    def train(self, neural_network, t, done, database, i_episode):
        # print('TRANSITION-MODEL: train')
        # Transition model training
        if (t % self.transition_model_buffer_sampling_rate == 0 and t != 0) or (
                self.train_end_episode and done):  # Sim pendulum: 200; mountain car: done TODO: check if use done
            # if (t == 500) or (self.train_end_episode and done)

            self._train_model_from_database(neural_network, database, i_episode, t)

    def get_state_representation(self, neural_network, observation, i_episode, t):


        self._preprocess_observation(np.array(observation))

        action_in = tf.ones([1, 1, 1], tf.float32)

        self.lstm_hidden_state = [tf.cast(tf.reshape(tf.zeros(self.lstm_h_size), [1, self.lstm_h_size]), tf.float32),
                                  tf.cast(tf.reshape(tf.zeros(self.lstm_h_size), [1, self.lstm_h_size]), tf.float32)]

        lstm_in = [tf.cast(tf.reshape(tf.zeros(150), [-1, 150]), tf.float32), tf.cast(tf.reshape(tf.zeros(150), [-1, 150]), tf.float32)]

        reshaped_network_input = tf.reshape(self.network_input[-1], [1, 1, 64, 64, 1])

        neural_network.model_parameters(batch_size=tf.constant(1),
                                        lstm_out_is_external = 1,
                                        lstm_out = self.lstm_hidden_state,
                                        lstm_in = lstm_in)

        if i_episode == 0 and t == 0:
            self.transition_model_predicting = neural_network.get_state_representation_model()



        #self.lstm_hidden_state_tensor = tf.convert_to_tensor(self.lstm_hidden_state, dtype=tf.float32)

        state_representation, ae_model_output = self.transition_model_predicting(
            [reshaped_network_input])

           # [self.network_input[-1]])
        '''
        print('concat2_parte2')
        print(concat2_parte2)
        print('concat_2')
        print(concat_2)
        print('x1')
        print(x1)
        print('olii')
        print('self.network_input[-1]')
        print(self.network_input[-1])
        '''

        self._refresh_image_plots(ae_model_output)  # refresh image plots
        self.t_counter += 1



        return state_representation

    def get_state_representation_batch(self, neural_network, observation_sequence_batch, action_sequence_batch,
                                       current_observation, buffer_length, i_episode, t):

        batch_size = len(observation_sequence_batch)
        # print('batch sizeee')
        # print(batch_size)

        if buffer_length == 20:
            # LSTM_HIDDEN_STATE_BATCH model
            neural_network.model_parameters(batch_size=batch_size,
                                            sequence_length=self.training_sequence_length)
            self.model_lstm_hidden_state_batch = neural_network.transition_model()

            print('Minimum buffer_length reached, Model "model_lstm_hidden_state_batch" created, give more feedback')

            # STATE REPRESENTATION BATCH model
            neural_network.model_parameters(batch_size=batch_size,
                                            sequence_length=tf.constant(1))
            self.model_state_representation_batch = neural_network.predicting_model()

            print('Minimum buffer_length reached, Model "model_state_representation_batch" created, give more feedback')

        # LSTM_HIDDEN_STATE_BATCH
        if i_episode != 0:
            self.model_lstm_hidden_state_batch.get_layer('conv1').set_weights(self.conv1_weights)
            self.model_lstm_hidden_state_batch.get_layer('norm_conv1').set_weights(self.norm_conv1_weights)
            self.model_lstm_hidden_state_batch.get_layer('conv2').set_weights(self.conv2_weights)
            self.model_lstm_hidden_state_batch.get_layer('norm_conv2').set_weights(self.norm_conv2_weights)
            self.model_lstm_hidden_state_batch.get_layer('conv3').set_weights(self.conv3_weights)

            self.model_lstm_hidden_state_batch.get_layer('fc_1').set_weights(self.fc_1_weights)
            self.model_lstm_hidden_state_batch.get_layer('fc_2').set_weights(self.fc_2_weights)
            self.model_lstm_hidden_state_batch.get_layer('rnn_layer').set_weights(self.rnn_layer_weights)

        transition_model_input = tf.convert_to_tensor(np.reshape(observation_sequence_batch,
                                                                 [batch_size * self.training_sequence_length,
                                                                  self.image_width, self.image_width, 1]),
                                                      dtype=tf.float32)
        action_in = tf.convert_to_tensor(
            np.reshape(action_sequence_batch, [batch_size * self.training_sequence_length, self.dim_a]),
            dtype=tf.float32)

        lstm_hidden_state_batch, _, _ = self.model_lstm_hidden_state_batch([transition_model_input, action_in])

        # FALTA COPIAR LOS WEIGHTS! como es el mismo modelo que el usado en el compute lstm hidden state, no ha hecho falta
        # definirlo dos veces en el fichero neural network, sino que haces otra duplica que admitira un batch size differente

        # STATE REPRESENTATION BATCH
        if i_episode != 0:
            self.model_state_representation_batch.get_layer('conv1').set_weights(self.conv1_weights)
            self.model_state_representation_batch.get_layer('norm_conv1').set_weights(self.norm_conv1_weights)
            self.model_state_representation_batch.get_layer('conv2').set_weights(self.conv2_weights)
            self.model_state_representation_batch.get_layer('norm_conv2').set_weights(self.norm_conv2_weights)
            self.model_state_representation_batch.get_layer('conv3').set_weights(self.conv3_weights)

            self.model_state_representation_batch.get_layer('fc_3').set_weights(self.fc_3_weights)
            self.model_state_representation_batch.get_layer('fc_4').set_weights(self.fc_4_weights)

            self.model_state_representation_batch.get_layer('deconv1').set_weights(self.deconv1_weights)
            self.model_state_representation_batch.get_layer('norm_deconv1').set_weights(self.norm_deconv1_weights)
            self.model_state_representation_batch.get_layer('deconv2').set_weights(self.deconv2_weights)
            self.model_state_representation_batch.get_layer('norm_deconv2').set_weights(self.norm_deconv2_weights)
            self.model_state_representation_batch.get_layer('deconv3').set_weights(self.deconv3_weights)

        transition_model_input = tf.convert_to_tensor(
            np.reshape(current_observation, [batch_size, self.image_width, self.image_width, 1]), dtype=tf.float32)

        state_representation_batch, _ = self.model_state_representation_batch(
            [transition_model_input, lstm_hidden_state_batch])

        self.conv1_weights_lstm_batch = self.model_lstm_hidden_state_batch.get_layer('conv1').get_weights()
        # print('weight conv1 lstm batch')
        # print(self.conv1_weights_lstm_batch[0][0][0][0][0])

        self.conv1_weights_state_batch = self.model_state_representation_batch.get_layer('conv1').get_weights()
        # print('weight conv1 state batch')
        # print(self.conv1_weights_state_batch[0][0][0][0][0])

        return state_representation_batch

    def compute_lstm_hidden_state(self, neural_network, action, i_episode, t):
        # print('FUNCTION: def compute_lstm_hidden_state')
        action = np.reshape(action, [1, self.dim_a])
        self.action_tensor = tf.convert_to_tensor(action, dtype=tf.float32)
        self.lstm_hidden_state_previous_time_step = self.lstm_hidden_state

        h_state, c_state = tf.split(self.lstm_hidden_state_previous_time_step, 2, axis=1)
        h_state = tf.reshape(h_state, 150)
        c_state = tf.reshape(c_state, 150)

        neural_network.model_parameters_compute_lstm(batch_size=tf.constant(1),
                                                     sequence_length=tf.constant(1),
                                                     h_state=h_state,
                                                     c_state=c_state)

        if i_episode == 0 and t == 0:
            self.model_compute_lstm_hidden_state = neural_network.compute_lstm_model()
            # print('CREO MODELO PREDICTINGG')

        if i_episode != 0 and t == 0:
            self.model_compute_lstm_hidden_state.get_layer('conv1').set_weights(self.conv1_weights)
            self.model_compute_lstm_hidden_state.get_layer('norm_conv1').set_weights(self.norm_conv1_weights)
            self.model_compute_lstm_hidden_state.get_layer('conv2').set_weights(self.conv2_weights)
            self.model_compute_lstm_hidden_state.get_layer('norm_conv2').set_weights(self.norm_conv2_weights)
            self.model_compute_lstm_hidden_state.get_layer('conv3').set_weights(self.conv3_weights)

            self.model_compute_lstm_hidden_state.get_layer('fc_1').set_weights(self.fc_1_weights)
            self.model_compute_lstm_hidden_state.get_layer('fc_2').set_weights(self.fc_2_weights)
            self.model_compute_lstm_hidden_state.get_layer('rnn_layer').set_weights(self.rnn_layer_weights)

        # self.lstm_hidden_state = self.model_compute_lstm_hidden_state([self.network_input[-1], self.action_tensor, self.lstm_hidden_state_previous_time_step])
        self.lstm_hidden_state = self.model_compute_lstm_hidden_state(
            [self.network_input[-1], self.action_tensor])

        self.last_actions.add(action)

        self.conv1_weights_lstm = self.model_compute_lstm_hidden_state.get_layer('conv1').get_weights()
        # print('weight conv1 lstm')
        # print(self.conv1_weights_lstm[0][0][0][0][0])

    def last_step(self, action_label):
        # print('FUNCTION: def last_step')
        if self.last_states.initialized() and self.last_actions.initialized():
            return [self.network_input[:-1],
                    self.last_actions.buffer[:-1],
                    self.network_input[-1],
                    action_label.reshape(self.dim_a)]
        else:
            return None

    def new_episode(self):
        # print('FUNCTION: def new_episode')
        self.lstm_hidden_state = np.zeros([1, 2 * self.lstm_h_size])
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 1]))

