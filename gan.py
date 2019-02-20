import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import sys

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9

x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1) # (60000, 784) instead of (60000, 28, 28)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

def get_batches(batch_size):
    """ Return batch_size of the x_train 
    vector at a time
    """
    current_index = 0
    while current_index + batch_size <= x_train.shape[0]:
        data_batch = x_train[current_index:current_index + batch_size]
        current_index += batch_size

        yield data_batch


def show_generator_output(sess, layer_count):
    """
    Show example output for the generator
    """
    input_real, input_z, t_value = model_inputs(z_dim, layer_count)
    
    example_z = np.random.uniform(-1, 1, size=[batch_size, z_dim])
    t_value_in = 1

    samples = sess.run(
        generator(input_z, layer_count, t_value, True),
        feed_dict={input_z: example_z, t_value: t_value_in})

    for sample in samples:
        plt.imshow(sample.reshape((layer_sizes[layer_count],layer_sizes[layer_count])), cmap=plt.cm.binary)
        plt.show()

def downsize_real(batch_images, layer_count):
    if layer_count == 2:
        return batch_images
    current_size = int(np.sqrt(batch_images.shape[1]))
    next_size = int(current_size / 2)
    new_arr = np.ndarray((batch_size, next_size**2))
    for i, image in enumerate(batch_images):
        image = image.reshape(current_size, current_size)
        re = image[:next_size*2, :next_size*2].reshape(next_size, 2, next_size, 2).max(axis=(1, 3)) #TODO: avg??
        re = re.reshape(next_size**2)
        new_arr[i] = re
    return downsize_real(new_arr, layer_count +1)

def scale_down_sample(image_tensor, layer_count):
    re = tf.reshape(image_tensor, (batch_size, layer_sizes[layer_count], layer_sizes[layer_count], 1))
    pool = tf.layers.average_pooling2d(re, [2,2], 2)
    back = tf.reshape(pool, (batch_size, layer_sizes[layer_count-1]**2))
    return back

def scale_up_sample(image_tensor, layer_count):
    re = tf.reshape(image_tensor, (batch_size, layer_sizes[layer_count-1],layer_sizes[layer_count-1], 1))
    near_neigh = tf.image.resize_nearest_neighbor(re, size=(layer_sizes[layer_count], layer_sizes[layer_count]))
    back = tf.reshape(near_neigh, [batch_size, layer_sizes[layer_count]**2])
    return back



def model_inputs(z_dim, layer_count):
    """
    Create the model input placeholders
    inputs_real is the tensor that holds vectors
    that represents the real images
    inputs_z is the tensor that holds the noise
    to be given to the generator 
    """
    image_size = layer_sizes[layer_count]**2

    inputs_real = tf.placeholder(tf.float32, shape=(None, image_size), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    t_value = tf.placeholder(tf.float32, (), name='t_value')
    
    return inputs_real, inputs_z, t_value

def discriminator(images, layer_count, t_value, reuse=False):
    """ The discriminator is the "art critic" 
    and will tell with a single node whether 
    the input given is a real (1) or fake (0)
    image. 
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        if layer_count ==  0:
            layer0 = tf.layers.dense(images, 1, name="d_lay0")
            out = tf.sigmoid(layer0)
            return out, layer0

        if layer_count == 1:
            layer1 = tf.layers.dense(images, layer_sizes[0]**2, activation=tf.nn.relu, name="d_lay1")
            down_sample = scale_down_sample(images, layer_count)
            layer1 = (1-t_value)*down_sample + t_value*layer1
            layer0 = tf.layers.dense(layer1, 1, name="d_lay0")
            out = tf.sigmoid(layer0)
            return out, layer0
        
        if layer_count == 2:
            layer2 = tf.layers.dense(images, layer_sizes[1]**2, activation=tf.nn.relu, name="d_lay2")
            down_sample = scale_down_sample(images, layer_count)
            layer2 = (1-t_value)*down_sample + t_value*layer2
            layer1 = tf.layers.dense(layer2, layer_sizes[0]**2, activation=tf.nn.relu, name="d_lay1")
            layer0 = tf.layers.dense(layer1, 1, name="d_lay0")
            out = tf.sigmoid(layer0)
            return out, layer0
        
        if layer_count == 3:
            layer3 = tf.layers.dense(images, layer_sizes[2]**2, activation=tf.nn.relu, name="d_lay3")
            down_sample = scale_down_sample(images, layer_count)
            layer3 = (1-t_value)*down_sample + t_value*layer3
            layer2 = tf.layers.dense(layer3, layer_sizes[1]**2, activation=tf.nn.relu, name="d_lay2")
            layer1 = tf.layers.dense(layer2, layer_sizes[0]**2, activation=tf.nn.relu, name="d_lay1")
            layer0 = tf.layers.dense(layer1, 1, name="d_lay0")
            out = tf.sigmoid(layer0)
            return out, layer0
        

def generator(z, layer_count, t_value, reuse=False):
    """ The generator will take a random noise 
    tensor z as input and generate a new never 
    seen before image.
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        layer0 = tf.layers.dense(z, layer_sizes[0]**2, activation=tf.nn.relu, name="g_lay0")
        if layer_count == 0:
            out = tf.tanh(layer0)
            return out

        layer1 = tf.layers.dense(layer0, layer_sizes[1]**2, activation=tf.nn.relu, name="g_lay1")
        if layer_count == 1:
            out = tf.tanh(layer1)
            up_sample = tf.tanh(scale_up_sample(layer0, layer_count))
            out = (1-t_value)*up_sample + t_value*out
            return out
        
        layer2 = tf.layers.dense(layer1, layer_sizes[2]**2, activation=tf.nn.relu, name="g_lay2")
        if layer_count == 2:
            out = tf.tanh(layer2)
            up_sample = tf.tanh(scale_up_sample(layer1, layer_count))
            out = (1-t_value)*up_sample + t_value*out
            return out
        
        
        layer3 = tf.layers.dense(layer2, layer_sizes[3]**2, activation=tf.nn.relu, name="g_lay3")
        if layer_count == 3:
            out = tf.tanh(layer3)
            up_sample = tf.tanh(scale_up_sample(layer2, layer_count))
            out = (1-t_value)*up_sample + t_value*out
            return out

def model_loss(input_real, input_z, layer_count, t_value):
    """
    Get the loss for the discriminator and generator
    """
    tf.print(input_z, output_stream=sys.stdout)
    label_smoothing = 0.9
    
    g_model = generator(input_z, layer_count, t_value)

    # For each run, the discriminator gets passed 2 inputs:
    # one real image and the fake image from the generator
    d_model_real, d_logits_real = discriminator(input_real, layer_count, t_value)
    d_model_fake, d_logits_fake = discriminator(g_model, layer_count, t_value, reuse=True)

    # The total loss of the discriminator is how well it classifies a real images as real (1)
    # plus how well it classifies a fake images as fake (0)
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_model_fake)))
    d_loss = d_loss_real + d_loss_fake

    # The loss for the generator is how well it manages to fool the discriminator 
    # that the generated image is not fake. This is the opposite of
    # d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake) * label_smoothing))

    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    """
    # Extract weights and biases for the two networks
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]
    
    for var in d_vars:
        print(var.name)

    # Optimize with AdamOptimizer, minimize the loss
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        with tf.variable_scope('optimize', reuse=tf.AUTO_REUSE):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, sess, layer_count, smoothing=False):
    """
    Train the GAN
    """

    input_real, input_z, t_value = model_inputs(z_dim, layer_count)
    d_loss, g_loss = model_loss(input_real, input_z, layer_count, t_value)
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    sess.run(tf.global_variables_initializer())
    steps = 0

    for epoch_i in range(epoch_count):
        print_loss = True
        for batch_images in get_batches(batch_size):
            downsized = downsize_real(batch_images, layer_count)
            # create some random noise
            batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
            t_value_in = 1
            if smoothing:
                t_value_in = steps/data_shape[0]
            # train the models on current batch
            _ = sess.run(d_opt, feed_dict={input_real: downsized, input_z: batch_z, t_value: t_value_in})
            _ = sess.run(g_opt, feed_dict={input_real: downsized, input_z: batch_z, t_value: t_value_in})

            steps += batch_size
            if steps % 6000 == 0:
                print(steps)
            
            if print_loss:
                # At the start of every epoch, get the losses and print them out
                print_loss = False
                train_loss_d = d_loss.eval({input_z: batch_z, input_real: downsized, t_value: 1}) #TODO: check with 1!!
                train_loss_g = g_loss.eval({input_z: batch_z, t_value: 1})

                print("Epoch {}/{}...".format(epoch_i+1, epochs),
                        "Discriminator Loss: {:.4f}...".format(train_loss_d),
                        "Generator Loss: {:.4f}".format(train_loss_g))
        
        # print("Done with one epoch")
        # show_generator_output(sess, layer_count)
        

batch_size = 16
z_dim = 100
learning_rate = 0.001
beta1 = 0 #0.5
epochs = 1
data_shape = 60000, 28*28
layer_sizes = [7, 14, 28]

if __name__ == "__main__":
    tf.enable_eager_execution()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            
            for layer_count in range(len(layer_sizes)):
                train(epochs, batch_size, z_dim, learning_rate, beta1, sess, layer_count, smoothing=True)
                print("Layed addition complete for layer_count: %d ... Starting regular training" % layer_count)
                train(epochs, batch_size, z_dim, learning_rate, beta1, sess, layer_count, smoothing=False)
            
            saver = tf.train.Saver()
            # Save the model to file
            save_path = saver.save(sess, './prog_model')
            print("model saved in %s" % save_path)
