import numpy as np
from tools.functions import observation_to_gray, FastImagePlot
from buffer import Buffer
import cv2
import tensorflow as tf

"""
Transition model
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
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]
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
        if self.resize_observation:
            observation = cv2.resize(observation, (self.image_width, self.image_width), interpolation=cv2.INTER_AREA)

        self.processed_observation = observation_to_gray(observation, self.image_width)
        self.last_states.add(self.processed_observation)
        self.network_input = np.array(self.last_states.buffer)

        self.network_input = tf.convert_to_tensor(self.network_input, dtype=tf.float32)

    def _refresh_image_plots(self, ae_model_output):
        if self.t_counter % 4 == 0 and self.show_observation:
            self.state_plot.refresh(self.processed_observation)

        if (self.t_counter + 2) % 4 == 0 and self.show_ae_output:
            self.ae_output_plot.refresh(ae_model_output)

    def _train_model_from_database(self, neural_network, database):
        episodes_num = len(database)

        print('Training Transition Model...')
        for i in range(self.number_training_iterations):  # Train
            if i % (self.number_training_iterations / 20) == 0:
                print('Progress Transition Model training: %i %%' % (i / self.number_training_iterations * 100))

            observations, actions, predictions = [], [], []

            # Sample batch from database
            for i in range(self.transition_model_sampling_size):
                count = 0
                while True:
                    count += 1
                    if count > 1000:  # check if it is possible to sample TODO: value should come from config
                        print('Database too small for training!')
                        return

                    selected_episode = round(np.random.uniform(-0.49, episodes_num - 1))  # select and episode from the database randomly
                    episode_trajectory_length = len(database[selected_episode])

                    if episode_trajectory_length > self.training_sequence_length + 2:
                        break

                sequence_start = round(np.random.uniform(0, episode_trajectory_length - self.training_sequence_length - 1))

                sequence = database[selected_episode][sequence_start:sequence_start + self.training_sequence_length + 1]  # get samples from database

                observation_seq = []
                action_seq = []

                # Separate observations, actions and expected observation predictions from sampled batch
                for i in range(len(sequence)):
                    observation_seq.append(sequence[i][0])
                    action_seq.append(sequence[i][1])

                observations.append(observation_seq[:-1])
                actions.append(action_seq[:-1])
                predictions.append(observation_seq[-1])

            observations = np.array(observations)  # these are required for LSTM training
            actions = np.array(actions)
            predictions = np.array(predictions)

            # Prepare inputs of NN
            action_in = np.reshape(actions, [self.transition_model_sampling_size,
                                             self.training_sequence_length,
                                             self.dim_a])

            transition_model_input = np.reshape(observations, [self.transition_model_sampling_size,
                                                               self.training_sequence_length,
                                                               self.image_width,
                                                               self.image_width,
                                                               1])

            lstm_hidden_state = [tf.zeros([self.transition_model_sampling_size, self.lstm_h_size]),
                                 tf.zeros([self.transition_model_sampling_size, self.lstm_h_size])]

            transition_model_label = np.reshape(predictions, [self.transition_model_sampling_size,
                                                              self.image_width,
                                                              self.image_width,
                                                              1]),

            NN_input = [transition_model_input,
                        tf.convert_to_tensor(action_in, dtype=tf.float32),
                        lstm_hidden_state]
            # Train
            self._compute_and_apply_grads(neural_network, NN_input, tf.convert_to_tensor(transition_model_label, dtype=tf.float32))

    @tf.function
    def _compute_and_apply_grads(self, neural_network, NN_input, transition_model_label):
        with tf.GradientTape() as tape_transition:
            _, _, prediction_value = self._eval_transition_model(neural_network, NN_input)
            loss = tf.reduce_mean(tf.square(prediction_value - transition_model_label))
            grads = tape_transition.gradient(loss, neural_network.NN_transition_model.trainable_variables)

        neural_network.transition_model_optimizer.apply_gradients(zip(grads, neural_network.NN_transition_model.trainable_variables))

    @tf.function
    def _eval_transition_model(self, neural_network, NN_input):
        lstm_hidden_state_batch, state_representation, transition_model_output = neural_network.NN_transition_model(NN_input)
        return lstm_hidden_state_batch, state_representation, transition_model_output

    def train(self, neural_network, t, done, database):
        # Transition model training
        if (t % self.transition_model_buffer_sampling_rate == 0 and t != 0) or (self.train_end_episode and done):  # Sim pendulum: 200; mountain car: done TODO: check if use done
            self._train_model_from_database(neural_network, database)

    def get_state_representation(self, neural_network, observation):
        self._preprocess_observation(np.array(observation))

        dummy_action_in = tf.ones([1, 1, 1], tf.float32)  # remove this somehow
        reshaped_network_input = tf.reshape(self.network_input[-1], [1, 1, self.image_width, self.image_width, 1])

        NN_input = [reshaped_network_input,
                    dummy_action_in,
                    self.lstm_hidden_state]

        _, state_representation, ae_model_output = self._eval_transition_model(neural_network, NN_input)

        self._refresh_image_plots(ae_model_output)  # refresh image plots
        self.t_counter += 1

        return state_representation

    def get_lstm_hidden_state_batch(self, neural_network, observation_sequence_batch, action_sequence_batch, batch_size):
        transition_model_input = tf.convert_to_tensor(np.reshape(observation_sequence_batch,
                                                                 [batch_size,
                                                                  self.training_sequence_length,
                                                                  self.image_width,
                                                                  self.image_width, 1]),
                                                      dtype=tf.float32)

        action_in = tf.convert_to_tensor(np.reshape(action_sequence_batch,
                                                    [batch_size,
                                                     self.training_sequence_length,
                                                     self.dim_a]),
                                         dtype=tf.float32)

        lstm_hidden_state_in = [tf.zeros([batch_size, self.lstm_h_size]),
                                tf.zeros([batch_size, self.lstm_h_size])]

        NN_input = [transition_model_input,
                    action_in,
                    lstm_hidden_state_in]

        lstm_hidden_state_batch,  _, _ = self._eval_transition_model(neural_network, NN_input)

        return lstm_hidden_state_batch

    def get_state_representation_batch(self, neural_network, current_observation, lstm_hidden_state_batch, batch_size):
        transition_model_input = tf.convert_to_tensor(np.reshape(current_observation, [batch_size,
                                                                                       1,
                                                                                       self.image_width,
                                                                                       self.image_width,
                                                                                       1]),
                                                       dtype=tf.float32)

        dummy_action_in = tf.ones([batch_size, 1, 1], tf.float32)  # TODO: something to do with this?

        NN_input = [transition_model_input,
                    dummy_action_in,
                    lstm_hidden_state_batch]

        _, state_representation_batch, _ = self._eval_transition_model(neural_network, NN_input)

        return state_representation_batch

    def compute_lstm_hidden_state(self, neural_network, action):
        action = np.reshape(action, [1, self.dim_a])
        reshaped_network_input = tf.reshape(self.network_input[-1], [1, 1, self.image_width, self.image_width, 1])

        NN_input = [reshaped_network_input,
                    action,
                    self.lstm_hidden_state]

        self.lstm_hidden_state, _, _ = self._eval_transition_model(neural_network, NN_input)

        self.last_actions.add(action)

    def last_step(self, action_label):
        if self.last_states.initialized() and self.last_actions.initialized():
            return [self.network_input[:-1],
                    self.last_actions.buffer[:-1],
                    self.network_input[-1],
                    action_label.reshape(self.dim_a)]
        else:
            return None

    def new_episode(self):
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 1]))