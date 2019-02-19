import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9



x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1) # (60000, 784) instead of (60000, 28, 28)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

def get_batches(batch_size):
    current_index = 0
    while current_index + batch_size <= x_train.shape[0]:
        data_batch = x_train[current_index:current_index + batch_size]
        current_index += batch_size

        yield data_batch

def model_inputs(image_size, z_dim):
    """
    Create the model inputs
    """
    inputs_real = tf.placeholder(tf.float32, shape=(None, image_size), name='input_real') 
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
    
    return inputs_real, inputs_z

def discriminator(images, reuse=False):
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
    with tf.variable_scope('generator', reuse=False if is_train==True else True):
        # Layer 1
        layer1 = tf.layers.dense(z, 128, activation=tf.nn.relu)
        
        # Layer 2
        layer2 = tf.layers.dense(layer1, 128, activation=tf.nn.relu)
        
        # Logits
        logits = tf.layers.dense(layer2, out_dim)
        
        out = tf.tanh(logits)
        
        return out

def model_loss(input_real, input_z, out_dim):
    """
    Get the loss for the discriminator and generator
    """
    label_smoothing = 0.9
    
    g_model = generator(input_z, out_dim)
    d_model_real, d_logits_real = discriminator(input_real)
    d_model_fake, d_logits_fake = discriminator(g_model, reuse=True)

    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                labels=tf.ones_like(d_model_real) * label_smoothing))
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.zeros_like(d_model_fake)))
    
    d_loss = d_loss_real + d_loss_fake

    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                labels=tf.ones_like(d_model_fake) * label_smoothing))

    return d_loss, g_loss

def model_opt(d_loss, g_loss, learning_rate, beta1):
    """
    Get optimization operations
    """
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
    g_vars = [var for var in t_vars if var.name.startswith('generator')]

    # Optimize
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)): 
        d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

    return d_train_opt, g_train_opt

def show_generator_output(sess, n_images, input_z, out_dim):
    """
    Show example output for the generator
    """
    z_dim = input_z.get_shape().as_list()[-1]
    example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

    samples = sess.run(
        generator(input_z, out_dim, False),
        feed_dict={input_z: example_z})

    plt.imshow(samples.reshape((28,28)), cmap=plt.cm.binary)
    plt.show()

def train(epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape):
    """
    Train the GAN
    """
    input_real, input_z = model_inputs(data_shape[1], z_dim)
    d_loss, g_loss = model_loss(input_real, input_z, data_shape[1])
    d_opt, g_opt = model_opt(d_loss, g_loss, learning_rate, beta1)
    
    steps = 0
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        try:
            saver.restore(sess, "./model.ckpt")
            print("Model restored!")
            for i in range(20):
                show_generator_output(sess, 1, input_z, data_shape[1])
            return
        except ValueError:
            sess.run(tf.global_variables_initializer())

        for epoch_i in range(epoch_count):
            print_loss = True
            for batch_images in get_batches(batch_size):
                
                steps += 1
            
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))
                
                _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                _ = sess.run(g_opt, feed_dict={input_real: batch_images, input_z: batch_z})
                
                if print_loss:
                    print_loss = False
                    # At the start of every epoch, get the losses and print them out
                    train_loss_d = d_loss.eval({input_z: batch_z, input_real: batch_images})
                    train_loss_g = g_loss.eval({input_z: batch_z})

                    print("Epoch {}/{}...".format(epoch_i+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))

        save_path = saver.save(sess, './model.ckpt')
        print("model saved in %s" % save_path)
        

batch_size = 16
z_dim = 10
learning_rate = 0.0002
beta1 = 0.5
epochs = 20
shape = 60000, 28*28

with tf.Graph().as_default():
    train(epochs, batch_size, z_dim, learning_rate, beta1, get_batches, shape)

    

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
# model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))   #10 because dataset is numbers from 0 - 9

# model.compile(optimizer='adam',  # Good default optimizer to start with
#               loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
#               metrics=['accuracy'])  # what to track

# model.fit(x_train, y_train, epochs=3)  # train the model

# val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
# print(val_loss)  # model's loss (error)
# print(val_acc)  # model's accuracy

# model.save('epic_num_reader.model')

# new_model = tf.keras.models.load_model('epic_num_reader.model')

# predictions = new_model.predict(x_test)

# print(np.argmax(predictions[0]))
# plt.imshow(x_test[0].reshape((28,28)), cmap=plt.cm.binary)
# plt.show()