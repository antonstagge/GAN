import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import numpy as np
import sys

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9

x_train = np.load('featured_extracted_data.npy')
print(x_train.shape)

dictionary = np.load('dictionary.npy')
# print("========== best 20")
# for i in range(20):
#     print(dictionary[-(i+1)])
# print()
# print("========= worst 20")
# for i in range(20):
#     print(dictionary[i])

# print(len(dictionary))
# exit()
# try not normalize?
# x_train = 2 * (x_train.reshape(x_train.shape[0], -1) / 255) - 1. # (60000, 784) instead of (60000, 28, 28)
# x_test = 2 * (x_test.reshape(x_test.shape[0], -1) / 255) - 1.

def save_review_sample(generated_samples, save_path='/fake_reviews/sample.txt'):
    f = open(save_path, 'wb+')
    for sample in generated_samples:
        sentence = ""
        for number in sample:
            index = int(number)
            if index < 0 or index > len(dictionary):
                sentence += " - "
            else:
                word = dictionary[index]
                sentence += " " + word
        
        sentence += '\n\n'
        f.write(sentence.encode('utf8'))
    f.close()

def save_visualization(X, nh_nw, save_path='./images/sample.jpg'):
    X = X.reshape(X.shape[0], 28, 28)
    h,w = X.shape[1], X.shape[2]
    img = np.zeros((h * nh_nw[0], w * nh_nw[1]))

    for n,x in enumerate(X):
        j = n // nh_nw[1]
        i = n % nh_nw[1]
        img[j*h:j*h+h, i*w:i*w+w] = x

    scipy.misc.imsave(save_path, img)

def lrelu(X, leak=0.2):
    f1 = 0.5 * (1 + leak) # 0.6
    f2 = 0.5 * (1 - leak) # 0.4
    return f1 * X + f2 * tf.abs(X)

def get_batches(batch_size, x_train):
    """ Return batch_size of the x_train 
    vector at a time
    """
    # # Shuffle training data
    index = np.arange(x_train.shape[0])
    np.random.shuffle(index)
    x_train = x_train[index]
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
        generator(input_z, out_dim),
        feed_dict={input_z: example_z})

    # samples = samples.reshape((n_images, 28,28))
    # samples = (samples + 1.) /2
    # samples = np.sqrt(samples)
    # for sample in samples:
    #     plt.imshow(sample, cmap=plt.cm.binary)
    #     plt.show()

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

def discriminator(images):
    """ The discriminator is the "art critic" 
    and will tell with a single node whether 
    the input given is a real (1) or fake (0)
    image. 
    """
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # Layer 1
        layer1 = lrelu(tf.layers.dense(images, 200))
        
        # Layer 2
        layer2 = lrelu(tf.layers.dense(layer1, 100))
        
        # Logits
        logits = lrelu(tf.layers.dense(layer2, 1))
        # Output
        out = tf.sigmoid(logits)
        
        return out, logits

def generator(z, out_dim):
    """ The generator will take a random noise 
    tensor z as input and generate a new never 
    seen before image.
    """
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        # Layer 1
        layer1 = lrelu(tf.layers.dense(z, 100))
        
        # Layer 2
        layer2 = lrelu(tf.layers.dense(layer1, 200))
        
        # Logits
        logits = tf.layers.dense(layer2, out_dim)
        
        # logits = tf.nn.tanh(logits)
        
        return logits

def model_loss(input_real, input_z, out_dim):
    """
    Get the loss for the discriminator and generator
    """
    label_smoothing = 0.95
    
    g_model = generator(input_z, out_dim)

    # For each run, the discriminator gets passed 2 inputs:
    # one real image and the fake image from the generator
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model)

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

def model_optimizers(d_loss, g_loss, learning_rate, beta1):
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

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, data_shape, show=False):
    """
    Train the GAN
    """
    input_real, input_z = model_inputs(data_shape[1], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[1])
    d_opt, g_opt = model_optimizers(d_loss, g_loss, learning_rate, beta1)
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            # raise ValueError
            # Try to restore a model from file
            saver.restore(sess, "./review_model.ckpt")
            print("Model restored!")
            if show:
                show_generator_output(sess, 10, input_z, data_shape[1])
                return
        except ValueError:
            # Init new model
            sess.run(tf.global_variables_initializer())

        
        for epoch_i in range(epoch_count):
            print_loss = True
            count = 0
            for batch_images in get_batches(batch_size, x_train):
                # create some random noise
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                count += batch_size
                # train the models on current batch
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})

                if count % 5000 == 0:
                    sys.stdout.write('.')
                    sys.stdout.flush()

                if print_loss:
                    # At the start of every epoch, get the losses and print them out
                    print_loss = False
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})
                    print()
                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

                    example_z = np.random.uniform(-1, 1, size=[16, z_dim])

                    generated_samples = sess.run(
                        generator(input_z, data_shape[1]),
                        feed_dict={input_z: example_z})
                    
                    save_review_sample(generated_samples[:10], save_path='./fake_reviews/sample_2_%03d.txt' % int(epoch_i))
                    # save_visualization(generated_samples, (4,4), save_path='./fake_images/sample_%03d.jpg' % int(epoch_i))
                    # save_visualization(batch_images, (4,4), save_path='./real_images/batch_%03d.jpg' % int(epoch_i))

        # Save the model to file
        save_path = saver.save(sess, './review_model.ckpt')
        print("model saved in %s" % save_path)
        

batch_size = 100
z_dim = 100
learning_rate = 0.001
beta1 = 0.2
epochs = 30
shape = x_train.shape

if __name__ == "__main__":
    with tf.Graph().as_default():
        train(epochs, batch_size, z_dim, learning_rate, beta1, shape, show=False)
