import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

z_dim = 10

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
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    
    return inputs_real, inputs_z, learning_rate

def discriminator(images, reuse=False):
    alpha = 0.2

    with tf.variable_scope('discriminator', reuse=reuse):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(1, activation=tf.nn.softmax))   #10 because dataset is numbers from 0 - 9

        return model



        
    

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

new_model = tf.keras.models.load_model('epic_num_reader.model')

predictions = new_model.predict(x_test)

print(np.argmax(predictions[0]))
plt.imshow(x_test[0].reshape((28,28)), cmap=plt.cm.binary)
plt.show()