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

    def model_parameters(self, batchsize, sequencelength, network_input_shape, lstm_hidden_state_shape, action_shape, lstm_hs_is_computed, autoencoder_mode):
        self.batchsize = batchsize
        self.sequencelength = sequencelength
        self.lstm_hs_is_computed = lstm_hs_is_computed

        self.network_input_shape = network_input_shape
        self.lstm_hidden_state_shape = lstm_hidden_state_shape
        self.action_shape = action_shape
        self.autoencoder_mode = autoencoder_mode

    def MyModel(self):
        batch_size = self.batchsize
        sequence_length = self.sequencelength
        '''
        #network_input_shape = self.network_input_shapeutshape
        #computedlstmboolean = self.computedlstmboolean
        print('batch_size: ')
        print(batch_size)
        '''

        #autoencoder_mode = True
        #inputShape = (self.image_width, self.image_width, 1)
        #inputShape = (20, 64, 64)


        transition_model_input = tf.keras.layers.Input(shape=(64, 64,1), batch_size=200)
        action_in = tf.keras.layers.Input(shape=(1), batch_size=200)
        computed_lstm_hs = tf.keras.layers.Input(shape=(300), batch_size=200)

        print('///////////////////////////')
        print(transition_model_input)
        print('action_in')
        print(action_in )
        '''
        #inputShape = self.network_input_shape
        print('print(inputShape)')
        print(inputShape)
        print(type(inputShape))
        '''

        lstm_hidden_state_size = 150
        '''
        print('self.lstm_hidden_state_shape')
        print(self.lstm_hidden_state_shape)
        '''


        # Convolutional encoder
       # transition_model_input = tf.keras.layers.Input(shape=inputShape)

        #action_in = tf.keras.layers.Input(shape=(100))
        #computed_lstm_hs = tf.keras.layers.Input(shape= self.lstm_hidden_state_shape)


       # computed_lstm_hs2 = tf.reshape(shape=[input.get_shape()[0].value, -1])(computed_lstm_hs)
        x = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')(transition_model_input)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        conv3 = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')(x)

        conv3_shape = conv3.get_shape()
        latent_space = tf.keras.layers.Flatten()(conv3)
        latent_space_shape = latent_space.get_shape()
        print('latent_space')
        print(latent_space)
        print('///////////////////////////')


        # Combine latent space information with actions from the policy
        fc_2 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh", name='fc_2')(latent_space)
        print('fc_2-latent space')
        print(fc_2)
        fc_1 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh", name='fc_1')(action_in)
        print('fc_1- action_in')
        print(fc_1)
        print('///////////////////////////')
        concat_1 = tf.concat([fc_1, fc_2], axis=1)
        print('concatdone')
        concat_1_shape = concat_1.get_shape()
        # Transform data into 3-D sequential structures: [batch size, sequence length, data size]
        sequential_concat_1 = tf.reshape(concat_1, shape=[batch_size, sequence_length, concat_1_shape[-1]])
        sequential_latent_space = tf.reshape(latent_space, shape=[batch_size, sequence_length, latent_space_shape[-1]])
        '''
        #print('sequential_latent_space')
        #print(sequential_latent_space)
        '''

        # LSTM

        my_LSTMCell = tf.keras.layers.LSTMCell(lstm_hidden_state_size)

        my_RNN_layer = tf.keras.layers.RNN(my_LSTMCell, return_sequences=True, return_state=True) # ver  que se obtiene por default si no especificas el tipo de return


       # lstm_hidden_state_out = my_RNN_layer(inputs = sequential_concat_1)
        whole_seq_output, final_memory_state, final_carry_state = my_RNN_layer(inputs=sequential_concat_1)
        '''
        #print(whole_seq_output.shape)

        #print(final_memory_state.shape)

        #print(final_carry_state.shape)
        '''

        lstm_hidden_state_out = final_memory_state

       # lstm_hidden_state = tf.cond(self.lstm_hs_is_computed, lambda:lstm_hidden_state_out , lambda: computed_lstm_hs)
        lstm_hidden_state = tf.cond(self.lstm_hs_is_computed, lambda:computed_lstm_hs , lambda: lstm_hidden_state_out)
        '''
        #print('computed_lstm_hs')
        #print(computed_lstm_hs)

        #print('lstm_hidden_state_out')
        #print(lstm_hidden_state_out)
        #print('lstm_hidden_state')
        #print(lstm_hidden_state)
        '''

        #lstm_hidden_state = lstm_hidden_state_out

        model_lstm_hidden_state = tf.keras.Model(inputs = [transition_model_input, action_in, computed_lstm_hs],  outputs =lstm_hidden_state)

        #concat_2 = tf.concat([lstm_hidden_state[:, -lstm_hidden_state_size:], sequential_latent_space[:, -1, :]], axis=1)
        concat_2 = tf.concat([lstm_hidden_state[:, -lstm_hidden_state_size:], sequential_latent_space[:, -1, :]],axis=1)
        # State representation
        state_representation = tf.keras.layers.Dense(1000, activation="tanh")(concat_2)
        model_state_representation = tf.keras.Model(inputs = [transition_model_input, action_in, computed_lstm_hs],  outputs =lstm_hidden_state)

        fc_4 = tf.keras.layers.Dense(latent_space_shape[1], activation="tanh")(state_representation)
        fc_4 = tf.reshape(fc_4, [-1, latent_space_shape[1]])
        fc_4 = tf.reshape(fc_4, [-1, conv3_shape[1], conv3_shape[2], conv3_shape[3]])  # go to shape of the latent space

        # Convolutional decoder
        dec_input = tf.cond(self.autoencoder_mode, lambda: fc_4, lambda: conv3)
        x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)
        


        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same', activation='sigmoid', name='deconv3')(x)

        # Model creation
        model_transition_model_output = tf.keras.Model(inputs=[transition_model_input, action_in],
                                                       outputs=transition_model_output)

        return model_lstm_hidden_state, model_state_representation, model_transition_model_output
