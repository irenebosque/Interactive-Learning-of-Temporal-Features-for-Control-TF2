import numpy as np
from tools.functions import observation_to_gray, FastImagePlot
from buffer import Buffer
import cv2
import tensorflow as tf

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
        self.lstm_hidden_state = np.zeros([1, 2 * self.lstm_h_size])
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
        print('TRANSITION-MODEL: _preprocess_observation')
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


    def _refresh_image_plots(self, neural_network, random_action):
        print('TRANSITION-MODEL: _refresh_image_plots')
        if self.t_counter % 4 == 0 and self.show_observation:
            self.state_plot.refresh(self.processed_observation)

        if (self.t_counter + 2) % 4 == 0 and self.show_ae_output:
            neural_network.model_parameters(batchsize_input_layer =tf.constant(1),batchsize=1, sequencelength=1, network_input_shape=self.network_input_shape,
                                            lstm_hidden_state_shape=self.lstm_hidden_state_shape,
                                            action_shape=self.action_shape,
                                            lstm_hs_is_computed=True, autoencoder_mode=True)

            ae_model_output = self.model_transition_model_output(
                [self.network_input[-1], self.random_action_tensor, self.lstm_hidden_state_tensor])


            self.ae_output_plot.refresh(ae_model_output)

    def _train_model_from_database(self, neural_network, database, random_action):
        print('TRANSITION-MODEL: _train_model_from_database')
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
            # Train transition model

            input_to_encoder = np.reshape(observations,
                                          [self.transition_model_sampling_size * self.training_sequence_length,
                                           self.image_width, self.image_width, 1])

            input_to_encoder = tf.convert_to_tensor(input_to_encoder, dtype=tf.float32)
            input_to_encoder_shape = tuple(input_to_encoder.get_shape().as_list())

            action_in = tf.convert_to_tensor(action_in, dtype=tf.float32)

            self.network_input_shape = tuple(self.network_input[-1].get_shape().as_list())
            self.lstm_hidden_state_shape = self.lstm_hidden_state.shape
            self.action_shape = action_in.shape

            neural_network.model_parameters(batchsize_input_layer = self.transition_model_sampling_size * self.training_sequence_length,
                                            batchsize=self.transition_model_sampling_size,
                                            sequencelength=self.training_sequence_length,
                                            network_input_shape=input_to_encoder_shape,
                                            lstm_hidden_state_shape=self.lstm_hidden_state_shape,
                                            action_shape=self.action_shape, lstm_hs_is_computed=False,
                                            autoencoder_mode=True)
            self.model_lstm_hidden_state, self.model_state_representation, self.model_transition_model_output = neural_network.MyModel()
            # Print model summary, notice the shape of the input layer
            #self.model_transition_model_output.summary()
            transition_model_label = np.reshape(predictions, [self.transition_model_sampling_size, self.image_width,
                                                              self.image_width, 1]),

            transition_model_label = tf.convert_to_tensor(transition_model_label, dtype=tf.float32)
            # save model as a png
            tf.keras.utils.plot_model(self.model_transition_model_output)

            # TRAIN
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)



            with tf.GradientTape() as tape:

                prediction_value = self.model_transition_model_output([input_to_encoder, action_in])
                current_loss = tf.reduce_mean(tf.square(prediction_value - transition_model_label))
                grads = tape.gradient(current_loss, self.model_transition_model_output.trainable_variables)

            optimizer.apply_gradients(zip(grads, self.model_transition_model_output.trainable_variables))

    def train(self, neural_network, t, done, database, random_action):
        print('TRANSITION-MODEL: train')
        # Transition model training
        if (t % self.transition_model_buffer_sampling_rate == 0 and t != 0) or (
                self.train_end_episode and done):  # Sim pendulum: 200; mountain car: done TODO: check if use done
            self._train_model_from_database(neural_network, database, random_action)

    def get_state_representation(self, neural_network, random_action, observation, bandera1):
        print('TRANSITION-MODEL: get_state_representation')
        self._preprocess_observation(np.array(observation))

        self.network_input_shape = tuple(self.network_input[-1].get_shape().as_list())
        self.lstm_hidden_state_shape = self.lstm_hidden_state.shape
        self.action_shape = random_action.shape  # IMPORTANT, in this function action WONT BE USED! but i write it to have all the inpputs defined

        neural_network.model_parameters(batchsize_input_layer =tf.constant(1),batchsize=tf.constant(1), sequencelength=tf.constant(1),
                                        network_input_shape=self.network_input_shape,
                                        lstm_hidden_state_shape=self.lstm_hidden_state_shape,
                                        action_shape=self.action_shape, lstm_hs_is_computed=tf.constant(True),
                                        autoencoder_mode=tf.constant(False))


        if bandera1 == 1:
            self.model_lstm_hidden_state, self.model_state_representation, self.model_transition_model_output = neural_network.MyModel()


        print(bandera1)
        #self.model_lstm_hidden_state, self.model_state_representation, self.model_transition_model_output = neural_network.MyModel()
        #self.model_state_representation.summary()
        self.random_action_tensor = tf.convert_to_tensor(random_action, dtype=tf.float32)
        self.lstm_hidden_state_tensor = tf.convert_to_tensor(self.lstm_hidden_state, dtype=tf.float32)


        state_representation = self.model_state_representation(
            [self.network_input[-1], self.random_action_tensor, self.lstm_hidden_state_tensor])

        self._refresh_image_plots(neural_network, random_action)  # refresh image plots
        self.t_counter += 1
        return state_representation

    def get_state_representation_batch(self, neural_network, observation_sequence_batch, action_sequence_batch,
                                       current_observation):

        batch_size = len(observation_sequence_batch)

        lstm_hidden_state_batch = neural_network.sess.run(neural_network.lstm_hidden_state,
                                                          feed_dict={
                                                              'transition_model/transition_model_input:0': np.reshape(
                                                                  observation_sequence_batch,
                                                                  [batch_batch_size, self.image_width, self.image_width,
                                                                   1])})

        state_representation_batch = neural_network.sess.run(neural_network.state_representation,
                                                             feed_dict={
                                                                 'transition_model/transition_model_input:0': np.reshape(
                                                                     current_observation,
                                                                     [batch_size, self.image_width, self.image_width,
                                                                      1])})

        return state_representation_batch

    def compute_lstm_hidden_state(self, neural_network, action):
        print('FUNCTION: def compute_lstm_hidden_state')
        action = np.reshape(action, [1, self.dim_a])

        neural_network.model_parameters(batchsize_input_layer =tf.constant(1),batchsize=tf.constant(1), sequencelength=tf.constant(1),
                                        network_input_shape=self.network_input_shape,
                                        lstm_hidden_state_shape=self.lstm_hidden_state_shape,
                                        action_shape=self.action_shape, lstm_hs_is_computed=tf.constant(False),
                                        autoencoder_mode=tf.constant(False))

        self.lstm_hidden_state = self.model_lstm_hidden_state(
            [self.network_input[-1], self.random_action_tensor, self.lstm_hidden_state_tensor])


        self.last_actions.add(action)

    def last_step(self, action_label):
        print('FUNCTION: def last_step')
        if self.last_states.initialized() and self.last_actions.initialized():
            return [self.network_input[:-1],
                    self.last_actions.buffer[:-1],
                    self.network_input[-1],
                    action_label.reshape(self.dim_a)]
        else:
            return None

    def new_episode(self):
        print('FUNCTION: def new_episode')
        self.lstm_hidden_state = np.zeros([1, 2 * self.lstm_h_size])
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 1]))