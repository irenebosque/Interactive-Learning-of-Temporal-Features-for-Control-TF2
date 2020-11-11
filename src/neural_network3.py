import tensorflow as tf
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

    def model_parameters(self, batch_size, lstm_out_is_external, lstm_out, lstm_in):
        self.batch_size = batch_size
        self.lstm_out_is_external = lstm_out_is_external
        self.lstm_out = lstm_out
        self.lstm_in = lstm_in



    def get_state_representation_model(self):
        transition_model_input = tf.keras.layers.Input(shape=(None, self.image_width, self.image_width, 1))

        LSTM_input = tf.keras.Input(shape=(None, 150))
        # Convolutional encoder
        conv1_layer = tf.keras.layers.Conv2D(16, (3, 3), strides=2, padding='same', name='conv1')
        conv2_layer = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')
        conv3_layer = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')
        x2 = tf.keras.layers.TimeDistributed(conv1_layer)(transition_model_input)

        #x1 = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')(transition_model_input)
        x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x2)
        x = tf.keras.layers.TimeDistributed(conv2_layer)(x)
        x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
        conv3 = tf.keras.layers.TimeDistributed(conv3_layer)(x)
        latent_space = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(conv3)

        latent_space_shape = latent_space.get_shape()



        lstm_out_external = tf.cast(tf.reshape(tf.zeros(150), [-1, 150]), tf.float32)


        final_memory_state = lstm_out_external

        concat2_parte1 = final_memory_state[:, -150:]
        print('concat2_parte1')
        print(concat2_parte1)
        concat2_parte2 = latent_space[:, -1, :]
        print('concat2_parte2')
        print(concat2_parte2)

        concat_2 = tf.concat([concat2_parte1, concat2_parte2], axis=1)


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
        # -----------------------------------------------------



        model_get_state_representation = tf.keras.Model(inputs=[transition_model_input],
                                                        outputs=[state_representation, transition_model_output],
                                        name="model_get_state_representation")
        



        tf.keras.utils.plot_model(model_get_state_representation ,
                                  to_file='model_get_state_representation .png',
                                  show_shapes=True,
                                  show_layer_names=True)

        #--------------------------------------------------------

        return model_get_state_representation



    def policy_model(self):

        # Inputs
        state_representation_input  = tf.keras.layers.Input(shape=(1000), batch_size=None)
        self.policy_input = tf.keras.layers.LayerNormalization()(state_representation_input)

        # Fully connected layers
        fc_5 = tf.keras.layers.Dense(1000, activation="relu", name='fc_5')(self.policy_input)
        fc_6 = tf.keras.layers.Dense(1000, activation="relu", name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(self.dim_a, activation="tanh", name='fc_7')(fc_6)
        self.policy_output = fc_7

        # Model creation
        model_policy = tf.keras.Model(inputs=[state_representation_input], outputs=[self.policy_output])
        return model_policy