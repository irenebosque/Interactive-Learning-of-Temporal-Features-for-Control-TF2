import tensorflow as tf
import tensorflow.contrib.layers as lays
import os

"""
Neural network structure.
"""


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):

        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

        # Build and load network if requested
        self._build_network()

        if load_transition_model:
            self._load_transition_model()

    def _build_network(self):  # check this
        with tf.variable_scope('transition_model'):
            # Create placeholders
            transition_model_input = tf.placeholder(tf.float32, (None, self.image_width, self.image_width, 1), name='transition_model_input')
            transition_model_label = tf.placeholder(tf.float32, (None, self.image_width, self.image_width, 1), name='transition_model_label')

            # Convolutional encoder
            conv1 = tf.contrib.layers.layer_norm(lays.conv2d(transition_model_input, 16, [3, 3], stride=2, padding='SAME'))
            conv2 = tf.contrib.layers.layer_norm(lays.conv2d(conv1, 8, [3, 3], stride=2, padding='SAME'))
            conv3 = lays.conv2d(conv2, 4, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)
            conv3_shape = conv3.get_shape()

            # Autoencoder latent space
            latent_space = tf.contrib.layers.flatten(conv3)
            self.state_representation = tf.identity(latent_space, name='state_representation')
            latent_space_shape = latent_space.get_shape()

            dec_input = conv3
            # Autoencoder decoder
            deconv1 = tf.contrib.layers.layer_norm(lays.conv2d_transpose(dec_input, 8, [3, 3], stride=2, padding='SAME'))
            deconv2 = tf.contrib.layers.layer_norm(lays.conv2d_transpose(deconv1, 16, [3, 3], stride=2, padding='SAME'))
            deconv3 = lays.conv2d_transpose(deconv2, 1, [3, 3], stride=2, padding='SAME', activation_fn=tf.nn.sigmoid)

            self.transition_model_output = tf.identity(deconv3, name='transition_model_output')

            # Prediction reconstruction loss
            reconstruction_loss = tf.reduce_mean(tf.square(self.transition_model_output - transition_model_label))

        # Autoencoder/Transition Model optimizer
        variables_transition_model = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'transition_model')
        self.train_transition_model = tf.train.AdamOptimizer(learning_rate=self.transition_model_learning_rate).minimize(reconstruction_loss, var_list=variables_transition_model)

        # Initialize tensorflow
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
        self.saver = tf.train.Saver(max_to_keep=100)
        self.saver_transition_model = tf.train.Saver(var_list=variables_transition_model)

    def save_transition_model(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)

        self.saver_transition_model.save(self.sess, self.network_loc + '_transition_model')

    def _load_transition_model(self):
        self.saver_transition_model.restore(self.sess, self.network_loc + '_transition_model')