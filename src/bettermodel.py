import tensorflow as tf
import numpy as np

#def compute_lstm_hidden_state
my_image = tf.ones([1, 1, 64, 64, 1], tf.float32)
my_action = tf.ones([1, 1, 1], tf.float32)
my_lstm_in = [[0, 0, 0],[0, 0, 0]]
lstm_out_is_external = 0

lstm_hidden_state_size = 150

transition_model_input = tf.keras.layers.Input(shape=(None, 64, 64, 1))
action_in = tf.keras.layers.Input(shape=(None, 1))
#lstm_in  = tf.keras.layers.Input(shape=(3, 3))



parte = tf.cast(tf.reshape(tf.zeros(150), [-1, 150]), tf.float32)
lstm_in = [parte, parte]
lstm_out = [parte, parte]

print('lstm_in')
print(lstm_in)


# Convolutional encoder
conv1_layer = tf.keras.layers.Conv2D(16, [3, 3], strides=2, padding='same', name='conv1')
conv2_layer = tf.keras.layers.Conv2D(8, [3, 3], strides=2, padding='same', name='conv2')
conv3_layer = tf.keras.layers.Conv2D(4, [3, 3], strides=2, padding='same', activation='sigmoid', name='conv3')
x = tf.keras.layers.TimeDistributed(conv1_layer)(transition_model_input)
x = tf.keras.layers.LayerNormalization(name='norm_conv1')(x)
x = tf.keras.layers.TimeDistributed(conv2_layer)(x)
x = tf.keras.layers.LayerNormalization(name='norm_conv2')(x)
conv3 = tf.keras.layers.TimeDistributed(conv3_layer)(x)
latent_space = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten(name='flatten'))(conv3)
model_encoder = tf.keras.Model(inputs=[transition_model_input], outputs=[latent_space], name='model_encoder')


latent_space_shape = latent_space.get_shape()

fc_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_1'))(action_in)
fc_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_2'))(latent_space)
concat_1 = tf.concat([fc_1, fc_2], axis=1, name='concat_1')
#-------------------------------------------------------------------------------------------------------

# LSTM (option 1)
my_LSTMCell = tf.keras.layers.LSTMCell(lstm_hidden_state_size)
my_RNN_layer = tf.keras.layers.RNN(my_LSTMCell, return_sequences=True, return_state=True, name='rnn_layer')
_, h_state, c_state = my_RNN_layer(inputs=concat_1, initial_state=lstm_in)

print('h_state')
print(h_state)
print(tf.shape(h_state))
#--------------------------------------------------------------------------

lstm_out_internal = h_state
lstm_out_external = lstm_out
model_lstm_hidden_state_out = tf.keras.Model(inputs=[transition_model_input, action_in],
                                             outputs=[lstm_out_internal], name='lstm_hidden_state_out')


final_memory_state = tf.cond(lstm_out_is_external == 1, lambda: lstm_out_external, lambda: lstm_out_internal)

#final_memory_state = np.array(final_memory_state)
print('final_memory_state')
print(final_memory_state)
print(tf.shape(final_memory_state))
concat2_parte1 = final_memory_state[:, -150:]
print('oppppp')
concat2_parte2 = latent_space[:, -1, :]
concat_2 = tf.concat([concat2_parte1, concat2_parte2], axis=1)




# State representation
state_representation = tf.keras.layers.Dense(1000, activation="tanh", name='fc_3')(concat_2)


fc_4 = tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_4')(state_representation)
fc_4 = tf.reshape(fc_4, [-1, 8, 8, 4] ) # go to shape of the latent space

# Convolutional decoder
dec_input = fc_4

x = tf.keras.layers.Conv2DTranspose(8, [3, 3], strides=2, padding='same', name='deconv1')(dec_input)

x = tf.keras.layers.LayerNormalization(name='norm_deconv1')(x)
x = tf.keras.layers.Conv2DTranspose(16, [3, 3], strides=2, padding='same', name='deconv2')(x)
x = tf.keras.layers.LayerNormalization(name='norm_deconv2')(x)
transition_model_output = tf.keras.layers.Conv2DTranspose(1, [3, 3], strides=2, padding='same',
                                                                  activation='sigmoid', name='deconv3')(x)
#-----------------------------------------------------


model_complete = tf.keras.Model(inputs=[transition_model_input, action_in], outputs=[transition_model_output], name="model_complete")
model_complete.summary()




tf.keras.utils.plot_model(model_complete,
                          to_file='simple_model.png',
                          show_shapes=True,
                          show_layer_names=True)

#--------------------------------------------------------------------------------



my_output_lstm_hidden_state_out = model_lstm_hidden_state_out([my_image, my_action])
print('my_output_lstm_hidden_state_out')
print(my_output_lstm_hidden_state_out)


##def _train_model_from_database
#my_image = tf.ones([20, 10, 64, 64, 1], tf.float32)
#my_action = tf.ones([20, 10, 1], tf.float32)
#my_lstm_in = [[0, 0, 0],[0, 0, 0]]

#my_output = model_complete([my_image, my_action])
#print('my_output')
#print(my_output)


