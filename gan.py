import tensorflow as tf
# import matplotlib.pyplot as plt
import numpy as np

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9

x_train = np.load('featured_extracted_data.npy')
print(x_train.shape)

dictionary = np.load('dictionary.npy')


# x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1) # (60000, 784) instead of (60000, 28, 28)
# x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

def get_batches(batch_size):
    """ Return batch_size of the x_train 
    vector at a time
    """
    current_index = 0
    while current_index + batch_size <= x_train.shape[0]:
        data_batch = x_train[current_index:current_index + batch_size]
        current_index += batch_size

        yield data_batch


def show_generator_output(sess, n_images, input_z, out_dim):
    """
    Show example output for the generator
    """
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_dim, False),
        feed_dict={input_z: example_z})

    sentence = ""
    for number in samples[0]:
        index = int(number)
        if index < 0 or index > len(dictionary):
            sentence += " - "
        else:
            word = dictionary[index]
            sentence += " " + word
    
    print(sentence)

def model_inputs(image_size, z_dim):
    """
    Create the model input placeholders
    inputs_real is the tensor that holds vectors
    that represents the real images
    inputs_z is the tensor that holds the noise
    to be given to the generator 
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_size), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z

def discriminator(images, reuse=False):
    """ The discriminator is the "art critic" 
    and will tell with a single node whether 
    the input given is a real (1) or fake (0)
    image. 
    """
    with tf.variable_scope('discriminator', reuse=reuse):
        # Layer 1
        layer1 = tf.layers.dense(images, 128, activation=tf.nn.relu)
        
        # Layer 2
        layer2 = tf.layers.dense(layer1, 128, activation=tf.nn.relu)
        
        # Logits
        logits = tf.layers.dense(layer2, 1)
        # Output
        out = tf.sigmoid(logits)
        
        return out, logits

def generator(z, out_dim, is_train=True):
    """ The generator will take a random noise 
    tensor z as input and generate a new never 
    seen before image.
    """
    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # Layer 1
        layer1 = tf.layers.dense(z, 128, activation=tf.nn.relu)
        
        # Layer 2
        layer2 = tf.layers.dense(layer1, 128, activation=tf.nn.relu)
        
        # Logits
        logits = tf.layers.dense(layer2, out_dim)
        
        # out = tf.math.round(logits)
        
        return logits

def model_loss(input_real, input_z, out_dim):
    """
    Get the loss for the discriminator and generator
    """
    label_smoothing = 0.9
    
    g_model = generator(input_z, out_dim)

    # For each run, the discriminator gets passed 2 inputs:
    # one real image and the fake image from the generator
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

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

    # Optimize with AdamOptimizer, minimize the loss
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, show=False):
    """
    Train the GAN
    """
    input_real, input_z = model_inputs(data_shape[1], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[1])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            # Try to restore a model from file
            saver.restore(sess, "./model.ckpt")
            print("Model restored!")
            if show:
                for i in range(1):
                    show_generator_output(sess, 1, input_z, data_shape[1])
                return
        except ValueError:
            # Init new model
            sess.run(tf.global_variables_initializer())

        count = 0
        for epoch_i in range(epoch_count):
            print_loss = True
            for batch_images in get_batches(batch_size):
                # create some random noise
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                # train the models on current batch
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                count += batch_size/data_shape[0]
                print("... %.3f" %count, end=' ')
                if print_loss:
                    # At the start of every epoch, get the losses and print them out
                    print_loss = False
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print()
                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

        # Save the model to file
        save_path = saver.save(sess, './model.ckpt')
        print("model saved in %s" % save_path)
        

batch_size = 16
z_dim = 100
learning_rate = 0.001
beta1 = 0.5
epochs = 20
shape = x_train.shape

if __name__ == "__main__":
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, shape, show=False)
