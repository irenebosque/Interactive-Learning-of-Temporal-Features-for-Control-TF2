import tensorflow as tf
import pandas as pd
import numpy as np

# Input layers
transition_model_input = tf.keras.layers.Input(shape=(None, 64, 64, 1))
action_in = tf.keras.layers.Input(shape=(None, 1))
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
latent_space_shape = latent_space.get_shape()

fc_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_1'))(action_in)
fc_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_space_shape[-1], activation="tanh", name='fc_2'))(latent_space)
concat_1 = tf.concat([fc_1, fc_2], axis=1, name='concat_1')
#-------------------------------------------------------------------------------------------------------

# LSTM (option 1)
my_LSTMCell = tf.keras.layers.LSTMCell(150)
my_RNN_layer = tf.keras.layers.RNN(my_LSTMCell, return_sequences=True, return_state=True, name='rnn_layer')
whole_seq_output, final_memory_state, final_carry_state = my_RNN_layer(inputs=concat_1)
#--------------------------------------------------------------------------

concat2_parte1 = final_memory_state[:, -150:]
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
output = transition_model_output


# Model creation
model = tf.keras.Model(inputs=[transition_model_input, action_in],
                                               outputs=[output],
                                               name='mymodel')

model.summary()
tf.keras.utils.plot_model(model,
                          to_file='simple_model.png',
                          show_shapes=True,
                          show_layer_names=True)

#--------------------------------------------------------------------------------

#Comprobando exitosamente que el mismo modelo admite diferentes batch y sequence lengths
my_image = tf.ones([20, 10, 64, 64, 1], tf.float32)
my_action = tf.ones([20, 10, 1], tf.float32)
my_output = model([my_image, my_action])
print('my_output')
print(my_output)

my_image2 = tf.ones([10, 5, 64, 64, 1], tf.float32)
my_action2 = tf.ones([10, 5, 1], tf.float32)
my_output2 = model([my_image2, my_action2])
#print('my_output2')
#print(my_output2)



