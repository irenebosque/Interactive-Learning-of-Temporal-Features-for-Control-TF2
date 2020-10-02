import tensorflow as tf
import numpy

# input_encoder = numpy.zeros((64,64), dtype=float)
class NeuralNetwork(tf.keras.Model):
    def __init__(self, latent_dim):
        super(NeuralNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.Densa = tf.keras.layers.Dense(self.latent_dim, activation='relu')
        '''
        # Autoencoder encoder
        self.conv1 = tf.keras.layers.Conv2D(16, [3, 3], stride=2, padding='same')
        # Autoencoder decoder
        self.deconv1 = tf.keras.layers.Conv2DTranspose(8, [3, 3], stride=2, padding='same')
        '''


    def call(self, x):
        '''
        x = self.conv1(x)
        x = self.deconv1(x)
        '''
        x = self.Densa(x)
        return x


neural_network = NeuralNetwork(latent_dim = 64)
print(neural_network)