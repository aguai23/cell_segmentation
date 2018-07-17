import tensorflow as tf
from net.ops import conv2d, fc, batch_norm
import logging
import numpy as np
from matplotlib import pyplot as plt
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class SimpleNet(object):

    def __init__(self, n_class=3, sample_size=51):
        tf.reset_default_graph()
        self.num_class = n_class
        self.sample_size = sample_size
        self.x = tf.placeholder(tf.float32, shape=[None, sample_size, sample_size, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, self.num_class])
        self.logits = self.build_graph(self.x)
        self.cost = self.get_cost(self.logits)
        self.session = None

    def build_graph(self, input, scope=""):

        with tf.name_scope(scope, "simple_net"):
            conv1 = conv2d(input, 32, [4, 4], scope="conv1", padding="VALID", weight_decay=0.1)
            conv1 = tf.nn.dropout(conv1, 0.9)
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="maxpool1")

            conv2 = conv2d(pool1, 64, [5, 5], padding="VALID", scope="conv2", weight_decay=0.1)
            conv2 = tf.nn.dropout(conv2, 0.8)
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="maxpool2")

            conv3 = conv2d(pool2, 128, [6, 6], scope="conv3", padding="VALID", weight_decay=0.1)
            conv3 = tf.nn.dropout(conv3, 0.75)
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="maxpool3")

            dims = pool3.get_shape()[1:]
            k = dims.num_elements()
            flatten_3 = tf.reshape(pool3, [-1, k])

            # k = pool1.get_shape()[1:].num_elements()
            # flatten_1 = tf.reshape(pool1, [-1, k])
            #
            # k = pool2.get_shape()[1:].num_elements()
            # flatten_2 = tf.reshape(pool2, [-1, k])
            #
            # fc1 = tf.concat([flatten_1, flatten_2, flatten_3], 1)
            fc1 = fc(flatten_3, 1024, scope="fc1", weight_decay=0.1)
            fc1 = tf.nn.dropout(fc1, 0.5)

            fc2 = fc(fc1, 1024, scope="fc2", weight_decay=0.1)
            fc2 = tf.nn.dropout(fc2, 0.5)

            output = fc(fc2, self.num_class, scope="output", activation=None)

            return output

    def residual_network(self, input, scope=""):

        with tf.name_scope(scope, "residual_net"):
            input = conv2d(input, 32, [1, 1], scope="dimension_increase1")
            conv1 = conv2d(input, 32, [8, 8], scope="conv1", padding="SAME", batch_norm_params={})
            conv2 = conv2d(conv1, 32, [8, 8], scope="conv2", padding="SAME", batch_norm_params={})
            output1 = conv2 + input
            pool1 = tf.nn.max_pool(output1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

            pool1 = conv2d(pool1, 64, [1, 1], scope="dimension_increase2")
            conv3 = conv2d(pool1, 64, [6, 6], scope="conv3", padding="SAME", batch_norm_params={})
            conv4 = conv2d(conv3, 64, [6, 6], scope="conv4", padding="SAME", batch_norm_params={})
            output2 = conv4 + pool1
            pool2 = tf.nn.max_pool(output2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

            pool2 = conv2d(pool2, 128, [1, 1], scope="dimension_increase3")
            conv5 = conv2d(pool2, 128, [3, 3], scope="conv5", padding="SAME", batch_norm_params={})
            conv6 = conv2d(conv5, 128, [3, 3], scope="conv6", padding="SAME", batch_norm_params={})
            output3 = conv6 + pool2
            pool3 = tf.nn.max_pool(output3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

            dims = pool3.get_shape()[1:]
            k = dims.num_elements()
            pool3 = tf.reshape(pool3, [-1, k])

            fc1 = fc(pool3, 1024, scope="fc1", weight_decay=0.1)
            fc1 = tf.nn.dropout(fc1, 0.5)

            fc2 = fc(fc1, 1024, scope="fc2", weight_decay=0.1)
            fc2 = tf.nn.dropout(fc2, 0.5)

            output = fc(fc2, self.num_class, scope="output", activation=None)

            return output

    def get_cost(self, logits):
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=self.y))
        # cost = tf.losses.mean_squared_error(self.y, tf.nn.softmax(logits))
        return cost

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint

        :param sess: current session
        :param model_path: path to file system location
        """
        print(model_path)
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def load_model(self, model_path):
        init = tf.global_variables_initializer()
        # initialize variables
        sess = tf.Session()
        sess.run(init)
        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        self.session = sess

    def predict(self, test_patch, label=None):
        if self.session:
            sess = self.session
            label_dummy = np.empty((test_patch.shape[0], self.num_class))
            if label is not None:
                cost, logits = sess.run([self.cost, self.logits],
                                        feed_dict={self.x: test_patch,
                                                   self.y: label})
                print(cost)
                if cost > 0.5:
                    plt.imshow(test_patch[0])
                    plt.show()
            else:
                logits = sess.run(self.logits,
                                  feed_dict={self.x: test_patch,
                                             self.y: label_dummy})
            exp_logits = np.exp(logits)
            prob = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

            return prob
        else:
            raise NotImplementedError("no model load")