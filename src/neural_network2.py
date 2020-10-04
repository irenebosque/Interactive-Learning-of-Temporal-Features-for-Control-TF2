import tensorflow as tf
import os

"""
Neural network structure.
"""


class NeuralNetwork(tf.keras.Model):
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):
        super(NeuralNetwork, self).__init__()
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

        # Autoencoder encoder
        self.conv1 = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')
        self.layer_norm_conv1 = tf.keras.layers.LayerNormalization()
        self.conv2 = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')
        self.layer_norm_conv2 = tf.keras.layers.LayerNormalization()
        self.conv3 = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')

        # Autoencoder decoder
        self.deconv1 = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')
        self.layer_norm_deconv1 = tf.keras.layers.LayerNormalization()
        self.deconv2 = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')
        self.layer_norm_deconv2 = tf.keras.layers.LayerNormalization()
        self.deconv3 = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same', activation='sigmoid', name='deconv3')


    def call(self, x):

        x = self.conv1(x)
        x = self.layer_norm_conv1(x)
        x = self.conv2(x)
        x = self.layer_norm_conv2(x)
        x = self.conv3(x)
        x = self.deconv1(x)
        x = self.layer_norm_deconv1(x)
        x = self.deconv2(x)
        x = self.layer_norm_deconv2(x)
        x = self.deconv3(x)
        transition_model_output = x
        return transition_model_output


    '''
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
    '''