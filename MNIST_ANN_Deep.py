'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

#from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import datetime
import time
# Parameters
learning_rate = 0.002
training_iters = 1001
display_step = 100
HiddenUnits=100
Net='in_5hid_out'
# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
batch_size = 100
stddev_VAR=0.5
SUB_FRAC=[1]#[256,128,64,32,16,8,4,2,1]
# tf Graph input

def getTimestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_')

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases):
    fc1 = tf.add(tf.matmul(x, weights['wd0']), biases['bd0'])
    fc1 = tf.nn.relu(fc1)
    #fc1=tf.nn.dropout(fc1,0.8)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.add(tf.matmul(fc1, weights['wd3']), biases['bd3'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.add(tf.matmul(fc1, weights['wd4']), biases['bd4'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.add(tf.matmul(fc1, weights['wd5']), biases['bd5'])
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return fc1



def main(_):
    mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
    timeSTP=getTimestamp()
    with open(timeSTP + "OutputANN_Deep.txt", "a") as text_file:
        text_file.write(
            "learningrate--{},stddev_VAR--{},network--{},HiddenUnits--{}".format(learning_rate,stddev_VAR,Net,HiddenUnits) + "\n")

    #start for-----------
    for frac in SUB_FRAC:
        x_train = np.array(mnist.train.images, dtype=np.float32)
        y_train = np.array(mnist.train.labels, dtype=np.int32)
        total_size = x_train.shape[0]
        pickList = np.random.randint(total_size, size=total_size // frac)
        x_train = x_train[pickList]
        y_train = y_train[pickList]
        n_batchs = total_size / float(batch_size)//frac
        # Store layers weight & bias
        weights = {
            'wd0': tf.Variable(tf.random_normal([28*28, HiddenUnits],stddev=stddev_VAR)),
            'wd1': tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits],stddev=stddev_VAR)),
            'wd2': tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR)),
            'wd3': tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR)),
            'wd4': tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR)),
            'wd5': tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR)),

            # 1024 inputs, 10 outputs (class prediction)
            'out': tf.Variable(tf.random_normal([HiddenUnits, n_classes],stddev=stddev_VAR))
        }

        biases = {
            'bd0': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'bd1': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'bd2': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'bd3': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'bd4': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'bd5': tf.Variable(tf.random_normal([HiddenUnits],stddev=stddev_VAR)),
            'out': tf.Variable(tf.random_normal([n_classes],stddev=stddev_VAR))
        }

        x = tf.placeholder(tf.float32,shape= [None, n_input])
        y = tf.placeholder(tf.float32, shape=[None, n_classes])

        # Construct model
        pred = conv_net(x, weights, biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # Initializing the variables


        # Launch the graph
        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        e = 0
        maxAcc=0
        # Keep training until reach max iterations
        while e  < training_iters:
            for b in range(int(n_batchs)):
                batch_ys = y_train[b * batch_size:(b + 1) * batch_size, :]
                batch_xs = x_train[b * batch_size:(b + 1) * batch_size, :]
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            accu=sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                y: mnist.test.labels})
            if accu > maxAcc:
                maxAcc=accu
            #if e % display_step == 0:
            print("Evaluate the accuracy on test dataset:")
            print("frac:{}, epoch: {}, accuracy: {}, current max accuracy: {}".format(frac, e, accu, maxAcc))
                #with open(timeSTP+"OutputANN_Deep.txt", "a") as text_file:
                    #text_file.write("frac--{}, epoch--{}, accuracy--{}, current max accuracy--{}".format(frac, e, accu,maxAcc) + "\n")
            e = e + 1
            #print(e)
        print("Optimization Finished!")

if __name__ == '__main__':
    tf.app.run()