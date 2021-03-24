import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf

"""
Implementation of HG-DAgger without uncertainty estimation
"""


class HG_DAGGER:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode):
        # Initialize variables
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode

        # Initialize HG_DAgger buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    @tf.function
    def _update_policy(self, neural_network, state_representation, policy_label):
        # Train policy model
        with tf.GradientTape() as tape_policy:
            policy_output = self._eval_action(neural_network, state_representation)
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - tf.cast(policy_label, dtype=tf.float32)))
            grads = tape_policy.gradient(policy_loss, neural_network.NN_policy.trainable_variables)

        neural_network.policy_optimizer.apply_gradients(zip(grads, neural_network.NN_policy.trainable_variables))

    @tf.function
    def _eval_action(self, neural_network, state_representation):
        action = neural_network.NN_policy(state_representation)
        return action

    def feed_h(self, h):
        self.h = np.reshape(h, [1, self.dim_a])

    def action(self, neural_network, state_representation):
        self.count += 1

        if np.any(self.h):  # if feedback, human teleoperates
            action = self.h
            print("feedback:", self.h[0])
        else:
            action = self._eval_action(neural_network, state_representation).numpy()

        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def train(self, neural_network, transition_model, action, t, done):
        # Add last step to memory buffer
        if transition_model.last_step(action) is not None and np.any(self.h):  # if human teleoperates, add action to database
            self.buffer.add(transition_model.last_step(action))

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and (t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
                action_sequence_batch = [np.array(pair[1]) for pair in batch]
                current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
                action_label_batch = [np.array(pair[3]) for pair in batch]
                batch_size = len(observation_sequence_batch)

                # Get batch of hidden states
                lstm_hidden_state_batch = transition_model.get_lstm_hidden_state_batch(neural_network,
                                                                                       observation_sequence_batch,
                                                                                       action_sequence_batch,
                                                                                       batch_size)

                # Compute state representation
                state_representation_batch = transition_model.get_state_representation_batch(neural_network,
                                                                                             current_observation_batch,
                                                                                             lstm_hidden_state_batch,
                                                                                             batch_size)

                self._update_policy(neural_network, state_representation_batch, action_label_batch)