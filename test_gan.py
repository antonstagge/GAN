import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import gan

with tf.Graph().as_default():
    with tf.Session() as sess:
        layer_count = 2

        input_real, input_z, t_value = gan.model_inputs(gan.z_dim, layer_count)
        d_loss, g_loss = gan.model_loss(input_real, input_z, layer_count, t_value)
        d_opt, g_opt = gan.model_opt(d_loss, g_loss, gan.learning_rate, gan.beta1)

        saver = tf.train.Saver()
        saver.restore(sess, './prog_model')

        gan.show_generator_output(sess, layer_count)