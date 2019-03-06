import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.distributions import Normal
from numpy.random import normal as gnr
import math
import smtplib
import datetime
import time
frommaddr = "labemailnliu03@gmail.com"
toaddr = "labemailnliu03@gmail.com"
message = "epoch--{}, accuracy--{}".format(10, "0.3")  # cant contain :
password = 'labemailnliu'

NUM_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
input_size = IMAGE_PIXELS
output_size = NUM_CLASSES
stddev_VAR=0.5
Net='in_5hid_out'
batch_size = 100
SUB_FRAC = [1] #[256,128,64,32,16,8,4,2,1] # randomly pick 1/SUB_FRAC of total data
n_epoch = 1001  # training epoch
n_samples = 10  # sampling number

hidden1_units = 100  # number of neurons in layer 1
hidden2_units = 100  # number of neurons in layer 2
HiddenUnits=100
learning_rate = 0.002  # learning rate
sigma_prior = np.exp(-3.0).astype(np.float32)

def getTimestamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S_')

def bnns(x,W_0, W_1, W_2, W_3,W_4,W_5,W_out,b_0, b_1, b_2, b_3,b_4,b_5,b_out):
    #in
    h = tf.nn.relu(tf.matmul(x, W_0) + b_0)
    h = tf.nn.relu(tf.matmul(h, W_1) + b_1)
    h = tf.nn.relu(tf.matmul(h, W_2) + b_2)
    h = tf.nn.relu(tf.matmul(h, W_3) + b_3)
    h = tf.nn.relu(tf.matmul(h, W_4) + b_4)
    h = tf.nn.relu(tf.matmul(h, W_5) + b_5)
    h = tf.nn.softmax(tf.matmul(h, W_out) + b_out)
    return h

def log_gaussin(x, mu, sigma):
    return (-0.5 * np.log(2. * np.pi) - tf.log(tf.abs(sigma)) - 0.5 * tf.square((x - mu) / sigma))


def log_gaussin_logsigma(x, mu, logsigma):
    return (-0.5 * np.log(2. * np.pi) - logsigma / 2. - tf.square(x - mu) / (2. * tf.exp(logsigma)))


def main(_):
    timeSTP = getTimestamp()
    # with open(timeSTP + "OutputBNN_Deep.txt", "a") as text_file:
    #     text_file.write(
    #         "learningrate--{},stddev_VAR--{},network--{},HiddenUnits--{}".format(learning_rate,stddev_VAR,Net,HiddenUnits) + "\n")
    mnist = input_data.read_data_sets("mnist/data", one_hot=True)
    for frac in SUB_FRAC:
        x_train = np.array(mnist.train.images, dtype=np.float32)
        y_train = np.array(mnist.train.labels, dtype=np.int32)
        total_size = x_train.shape[0]
        # create sub data set
        pickList = np.random.randint(total_size, size=total_size// frac )
        x_train = x_train[pickList]
        y_train = y_train[pickList]
        n_batchs = total_size / float(batch_size)//frac
        # MODEL
        #in
        w0_mu = tf.Variable(tf.random_normal([input_size, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w0_logsigma = tf.Variable(tf.random_normal([input_size, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b0_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b0_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)

        # 1L
        w1_mu = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w1_logsigma = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b1_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b1_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)

        # 2L
        w2_mu = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w2_logsigma = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b2_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b2_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)

        # 3L
        w3_mu = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w3_logsigma = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b3_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b3_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)

        # 4L
        w4_mu = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w4_logsigma = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b4_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b4_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)

        # 5L
        w5_mu = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        w5_logsigma = tf.Variable(tf.random_normal([HiddenUnits, HiddenUnits], stddev=stddev_VAR), trainable=True)
        b5_mu = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        b5_logsigma = tf.Variable(tf.random_normal([HiddenUnits], stddev=stddev_VAR), trainable=True)
        # out
        wout_mu = tf.Variable(tf.random_normal([HiddenUnits, output_size], stddev=stddev_VAR), trainable=True)
        wout_logsigma = tf.Variable(tf.random_normal([HiddenUnits, output_size], stddev=stddev_VAR), trainable=True)
        bout_mu = tf.Variable(tf.random_normal([output_size], stddev=stddev_VAR), trainable=True)
        bout_logsigma = tf.Variable(tf.random_normal([output_size], stddev=stddev_VAR), trainable=True)

        x_ph = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS])
        y_ph = tf.placeholder(tf.float32, shape=[None, NUM_CLASSES])

        sample_log_z = [0.0] * n_samples
        sample_log_qz = [0.0] * n_samples
        sample_log_likelihood = [0.0] * n_samples
        for s in range(n_samples):
            #in
            start = time.time()

            epsilon_w0 = tf.random_normal([input_size, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b0 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w0 = w0_mu + tf.nn.relu(w0_logsigma) * epsilon_w0
            b0 = b0_mu + tf.nn.relu(b0_logsigma) * epsilon_b0

            epsilon_w1 = tf.random_normal([HiddenUnits, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b1 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w1 = w1_mu + tf.nn.relu(w1_logsigma) * epsilon_w1
            b1 = b1_mu + tf.nn.relu(b1_logsigma) * epsilon_b1


            epsilon_w2 = tf.random_normal([HiddenUnits, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b2 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w2 = w2_mu + tf.nn.relu(w2_logsigma) * epsilon_w2
            b2 = b2_mu + tf.nn.relu(b2_logsigma) * epsilon_b2

            epsilon_w3 = tf.random_normal([HiddenUnits, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b3 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w3 = w3_mu + tf.nn.relu(w3_logsigma) * epsilon_w3
            b3 = b3_mu + tf.nn.relu(b3_logsigma) * epsilon_b3

            epsilon_w4 = tf.random_normal([HiddenUnits, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b4 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w4 = w4_mu + tf.nn.relu(w4_logsigma) * epsilon_w4
            b4 = b4_mu + tf.nn.relu(b4_logsigma) * epsilon_b4

            epsilon_w5 = tf.random_normal([HiddenUnits, HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            epsilon_b5 = tf.random_normal([HiddenUnits], stddev=sigma_prior, dtype=tf.float32)
            w5 = w5_mu + tf.nn.relu(w5_logsigma) * epsilon_w5
            b5 = b5_mu + tf.nn.relu(b5_logsigma) * epsilon_b5


            epsilon_wout = tf.random_normal([HiddenUnits, output_size], stddev=sigma_prior, dtype=tf.float32)
            epsilon_bout = tf.random_normal([output_size], stddev=sigma_prior, dtype=tf.float32)
            wout = wout_mu + tf.nn.relu(wout_logsigma) * epsilon_wout
            bout = bout_mu + tf.nn.relu(bout_logsigma) * epsilon_bout

            y = bnns(x_ph,w0, w1, w2, w3, w4, w5,wout, b0, b1, b2, b3, b4, b5,bout)
            elapsed = time.time() - start
            print('elapsedSample:' + str(elapsed))
            # prior distribution p(z)
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w0, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b0, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w1, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b1, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w2, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b2, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w3, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b3, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w4, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b4, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(w5, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(b5, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(wout, 0.0, sigma_prior))
            sample_log_z[s] += tf.reduce_sum(log_gaussin(bout, 0.0, sigma_prior))

            # variational posterior distribution q(z|\theta)
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w0, w0_mu, w0_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b0, b0_mu, b0_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w1, w1_mu, w1_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b1, b1_mu, b1_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w2, w2_mu, w2_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b2, b2_mu, b2_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w3, w3_mu, w3_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b3, b3_mu, b3_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w4, w4_mu, w4_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b4, b4_mu, b4_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(w5, w5_mu, w5_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(b5, b5_mu, b5_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(wout, wout_mu, wout_logsigma * 2.0))
            sample_log_qz[s] += tf.reduce_sum(log_gaussin_logsigma(bout, bout_mu, bout_logsigma * 2.0))

            # likelihood p(D|z)
            sample_log_likelihood[s] = tf.reduce_sum(log_gaussin(y, y_ph, sigma_prior))

        log_z = tf.reduce_sum(sample_log_z) / n_samples
        log_qz = tf.reduce_sum(sample_log_qz) / n_samples
        log_likelihood = tf.reduce_sum(sample_log_likelihood) / n_samples
        loss = tf.reduce_sum(1.0 / n_batchs * (log_qz - log_z) - log_likelihood) / float(batch_size)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        y_mu = bnns(x_ph, w0_mu,w1_mu, w2_mu, w3_mu,w4_mu,w5_mu,wout_mu, b0_mu,b1_mu, b2_mu, b3_mu,b4_mu,b5_mu,bout_mu)
        correct_prediction = tf.equal(tf.argmax(y_mu, 1), tf.argmax(y_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # End of Constructing the computing graph

        sess = tf.InteractiveSession()
        # Train
        tf.global_variables_initializer().run()

        # writer = tf.train.SummaryWriter("./test",sess.graph)
        maxAcc = 0
        for e in range(n_epoch):

            errs = []
            for b in range(int(n_batchs)):
                batch_xs = x_train[b * batch_size:(b + 1) * batch_size, :]
                batch_ys = y_train[b * batch_size:(b + 1) * batch_size, :]
                _, ll, epsl_1, epsl_2, epsl_3 = sess.run([train_step, loss, log_qz, log_z, log_likelihood],
                                                         feed_dict={x_ph: batch_xs, y_ph: batch_ys})
                errs.append(ll)
            start2 = time.perf_counter()
            accu, y_pred = sess.run([accuracy, y_mu], feed_dict={x_ph: mnist.test.images, y_ph: mnist.test.labels})
            elapsed2 = time.perf_counter()-start2
            #print("bnnElapased:"+str(bnnElapased))
            if accu > maxAcc:
                maxAcc = accu
            #if e % 100 == 0:
            print("Evaluate the accuracy on test dataset:")
            print("frac:{}, epoch: {}, accuracy: {}, current max accuracy: {}".format(frac, e, accu, maxAcc))
            print('\n')
            print(elapsed)
            print('\n')
            print(elapsed2)
                # server = smtplib.SMTP('smtp.gmail.com:587')
                # server.starttls()
                # server.login(frommaddr, password)
                # server.sendmail(frommaddr, toaddr,
                #                 "frac--{},epoch--{}, accuracy--{}, current max accuracy--{}".format(frac,e, accu, maxAcc))
                # server.quit()
                #with open(timeSTP+"OutputBNN_Deep.txt", "a") as text_file:
                    #text_file.write("frac--{}, epoch--{}, accuracy--{}, current max accuracy--{}".format(frac, e, accu,
                                                                                                         #maxAcc) + "\n")


if __name__ == "__main__":
    tf.app.run()
