import tensorflow as tf


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_transition_model, load_policy, dim_a, network_loc, image_size):
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate

    def model_parameters(self, batchsize_input_layer, batchsize, sequencelength, network_input_shape,
                         lstm_hidden_state_shape, action_shape,
                         lstm_hs_is_computed):
        self.batchsize_input_layer = batchsize_input_layer
        self.batchsize = batchsize
        self.sequencelength = sequencelength
        self.lstm_hs_is_computed = lstm_hs_is_computed

        self.network_input_shape = network_input_shape
        self.lstm_hidden_state_shape = lstm_hidden_state_shape
        self.action_shape = action_shape


    def MyModel(self):
        batch_size = self.batchsize
        sequence_length = self.sequencelength

        transition_model_input = tf.keras.layers.Input(shape=(64, 64, 1), batch_size=self.batchsize_input_layer)
        action_in = tf.keras.layers.Input(shape=(1), batch_size=self.batchsize_input_layer)
        computed_lstm_hs = tf.keras.layers.Input(shape=(300), batch_size=self.batchsize_input_layer)

        lstm_hidden_state_size = 150

        # Convolutional encoder

        x = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')(transition_model_input)
        x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x)
        x = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
        conv3 = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')(x)

        conv3_shape = conv3.get_shape()
        latent_space = tf.keras.layers.Flatten()(conv3)
        latent_space_shape = latent_space.get_shape()

        # Combine latent space information with actions from the policy
        fc_2 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh", name='fc_2')(latent_space)
        fc_1 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh", name='fc_1')(action_in)

        concat_1 = tf.concat([fc_1, fc_2], axis=1)
        concat_1_shape = concat_1.get_shape()

        # Transform data into 3-D sequential structures: [batch size, sequence length, data size]
        sequential_concat_1 = tf.reshape(concat_1, shape=[batch_size, sequence_length, concat_1_shape[-1]])
        sequential_latent_space = tf.reshape(latent_space, shape=[batch_size, sequence_length, latent_space_shape[-1]])

        # LSTM

        my_LSTMCell = tf.keras.layers.LSTMCell(lstm_hidden_state_size)
        my_RNN_layer = tf.keras.layers.RNN(my_LSTMCell, return_sequences=True,
                                           return_state=True)

        whole_seq_output, final_memory_state, final_carry_state = my_RNN_layer(inputs=sequential_concat_1)

        lstm_hidden_state_out = final_memory_state
        #lstm_hidden_state = tf.cond(self.lstm_hs_is_computed, lambda: computed_lstm_hs, lambda: lstm_hidden_state_out)
        lstm_hidden_state = lstm_hidden_state_out


        concat_2 = tf.concat([lstm_hidden_state[:, -lstm_hidden_state_size:], sequential_latent_space[:, -1, :]],
                             axis=1)
        # State representation
        state_representation = tf.keras.layers.Dense(1000, activation="tanh")(concat_2)

        fc_4 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh")(state_representation)
        fc_4 = tf.reshape(fc_4, [-1, latent_space_shape[1]])
        fc_4 = tf.reshape(fc_4, [-1, conv3_shape[1], conv3_shape[2], conv3_shape[3]])  # go to shape of the latent space

        # Convolutional decoder
        dec_input = fc_4
        x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)

        x = tf.keras.layers.LayerNormalization(name='norm_deconv1')(x)
        x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv2')(x)
        transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same',
                                                                  activation='sigmoid', name='deconv3')(x)



        # Model creation
        model_transition_model_output = tf.keras.Model(inputs=[transition_model_input, action_in],
                                                       outputs=[lstm_hidden_state_out, state_representation,transition_model_output])

        return model_transition_model_output


    def predicting_model(self):
        batch_size = self.batchsize
        sequence_length = self.sequencelength

        transition_model_input = tf.keras.layers.Input(shape=(64, 64, 1), batch_size=self.batchsize_input_layer)
        computed_lstm_hs = tf.keras.layers.Input(shape=(300), batch_size=self.batchsize_input_layer)

        lstm_hidden_state_size = 150

        # Convolutional encoder
        x = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')(transition_model_input)
        x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x)
        x = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
        conv3 = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')(x)
        conv3_shape = conv3.get_shape()

        latent_space = tf.keras.layers.Flatten()(conv3)
        latent_space_shape = latent_space.get_shape()

        # Transform data into 3-D sequential structures: [batch size, sequence length, data size]

        sequential_latent_space = tf.reshape(latent_space, shape=[batch_size, sequence_length, latent_space_shape[-1]])
        lstm_hidden_state = computed_lstm_hs
        concat_2 = tf.concat([lstm_hidden_state[:, -lstm_hidden_state_size:], sequential_latent_space[:, -1, :]],axis=1)

        # State representation
        state_representation = tf.keras.layers.Dense(1000, activation="tanh")(concat_2)

        fc_4 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh")(state_representation)
        fc_4 = tf.reshape(fc_4, [-1, latent_space_shape[1]])
        fc_4 = tf.reshape(fc_4, [-1, conv3_shape[1], conv3_shape[2], conv3_shape[3]])  # go to shape of the latent space

        # Convolutional decoder
        dec_input = fc_4
        x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)

        x = tf.keras.layers.LayerNormalization(name='norm_deconv1')(x)
        x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
        x = tf.keras.layers.LayerNormalization(name='norm_deconv2')(x)
        transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same',
                                                                  activation='sigmoid', name='deconv3')(x)

        # Model creation
        predicting_model = tf.keras.Model(inputs=[transition_model_input, computed_lstm_hs],
                                                       outputs=[state_representation,transition_model_output])

        return predicting_model

    def my_policy(self):

        # Inputs
        state_representation_input  = tf.keras.layers.Input(shape=(1000), batch_size=self.batchsize_input_layer)
        self.policy_input = tf.keras.layers.LayerNormalization()(state_representation_input)

        # Fully connected layers
        fc_5 = tf.keras.layers.Dense(1000, activation="relu", name='fc_5')(self.policy_input)
        fc_6 = tf.keras.layers.Dense(1000, activation="relu", name='fc_6')(fc_5)
        fc_7 = tf.keras.layers.Dense(self.dim_a, activation="tanh", name='fc_7')(fc_6)
        self.policy_output = fc_7

        # Model creation
        model_policy = tf.keras.Model(inputs=[state_representation_input], outputs=[self.policy_output])
        return model_policy

