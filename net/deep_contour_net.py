import tensorflow as tf
from net.ops import conv2d
from tensorflow.contrib.layers import conv2d_transpose
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class DeepContourNet(object):

    def __init__(self, n_class=2, cost="cross_entropy", sample_size=224):
        tf.reset_default_graph()
        self.n_class = n_class
        self.x = tf.placeholder("float", shape=[None, sample_size, sample_size, 3])
        self.mask = tf.placeholder("float", shape=[None, sample_size, sample_size, 2])
        self.contour_mask = tf.placeholder("float", shape=[None, sample_size, sample_size, 2])
        self.output_mask, self.output_contour = self.build_net(self.x)
        self.cost = self.get_cost(self.output_mask, self.output_contour, cost)
        self.session = None

    def build_net(self, input, scope=""):

        with tf.name_scope(scope, "dcnn"):
            # down sample
            conv0 = conv2d(input, 64, [3, 3], scope="conv0", padding="SAME", batch_norm_params={})
            conv0_out = conv2d(conv0, 64, [3, 3], scope="conv0_out", padding="SAME", batch_norm_params={})
            maxpool0 = tf.nn.max_pool(conv0_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                      name="maxpool0")

            conv1 = conv2d(maxpool0, 128, [3, 3], scope="conv1", padding="SAME", batch_norm_params={})
            conv1_out = conv2d(conv1, 128, [3, 3], scope="conv1_out", padding="SAME", batch_norm_params={})
            maxpool1 = tf.nn.max_pool(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                      name="maxpool1")

            conv2 = conv2d(maxpool1, 256, [3, 3], scope="conv2", padding="SAME", batch_norm_params={})
            conv2_out = conv2d(conv2, 256, [3, 3], scope="conv2_out", padding="SAME", batch_norm_params={})

            # up sample
            deconv0 = conv2d_transpose(conv2_out, 128, [2, 2], [2, 2], padding="SAME", activation_fn=tf.nn.relu,
                                       weights_regularizer=tf.nn.l2_loss)
            deconv0_concat = tf.concat([deconv0, conv1_out], 3)
            deconv0_out = conv2d(deconv0_concat, 128, [3, 3], scope="deconv0_out", padding="SAME", batch_norm_params={})

            deconv1 = conv2d_transpose(deconv0_out, 128, [2, 2], [2, 2], padding="SAME", activation_fn=tf.nn.relu)
            deconv1_concat = tf.concat([deconv1, conv0_out], 3)
            deconv1_out = conv2d(deconv1_concat, 64, [3, 3], scope="deconv1_out", padding="SAME", batch_norm_params={})

            deconv_contour0 = conv2d_transpose(conv2_out, 128, [2, 2], [2, 2], padding="SAME", activation_fn=tf.nn.relu,
                                               weights_regularizer=tf.nn.l2_loss)
            deconv_contour0_concat = tf.concat([deconv_contour0, conv1_out], 3)
            deconv_contour0_out = conv2d(deconv_contour0_concat, 128, [3, 3], scope="deconv_contour0_out",
                                         padding="SAME", batch_norm_params={})

            deconv_contour1 = conv2d_transpose(deconv_contour0_out, 64, [2, 2], [2, 2], padding="SAME",
                                               activation_fn=tf.nn.relu)
            deconv_contour1_concat = tf.concat([deconv_contour1, conv0_out], 3)
            deconv_contour1_out = conv2d(deconv_contour1_concat, 128, [3, 3], scope="deconv_contour1_out",
                                         padding="SAME", batch_norm_params={})

            output_mask = conv2d(deconv1_out, 2, [3, 3], scope="output_mask", padding="SAME")
            output_contour = conv2d(deconv_contour1_out, 2, [3, 3], scope="output_contour", padding="SAME")

            return output_mask, output_contour

    def get_cost(self, output_map, output_contour, cost_name):
        flat_output_map = tf.reshape(output_map, [-1, self.n_class])
        flat_output_contour = tf.reshape(output_contour, [-1, self.n_class])
        flat_mask = tf.reshape(self.mask, [-1, self.n_class])
        flat_contour = tf.reshape(self.contour_mask, [-1, self.n_class])

        if cost_name == "cross_entropy":
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_map,
                                                                          labels=flat_mask)) + \
                   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_contour,
                                                                          labels=flat_contour))
        else:
            raise ValueError("unknown cost function")

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

    def predict(self, x_test, mask=None, contour=None):
        if self.session:
            sess = self.session
            mask_dummy = np.empty(list(x_test.shape[0:3]) + [self.n_class])
            contour_dummy = np.empty(list(x_test.shape[0:3]) + [self.n_class])
            if mask is not None and contour is not None:
                cost, output_mask, output_contour = sess.run([self.cost, self.output_mask, self.output_contour],
                                                             feed_dict={self.x: x_test,
                                                                        self.mask: mask,
                                                                        self.contour_mask: contour})
                print(cost)
            else:
                output_mask, output_contour = sess.run([self.output_mask, self.output_contour],
                                                       feed_dict={self.x: x_test,
                                                                  self.mask: mask_dummy,
                                                                  self.contour_mask: contour_dummy})

            output_mask = np.divide(np.exp(output_mask), np.sum(np.exp(output_mask), axis=3, keepdims=True))
            output_contour = np.divide(np.exp(output_contour), np.sum(np.exp(output_contour), axis=3, keepdims=True))
            output_mask = np.argmax(output_mask, axis=3)
            output_contour = np.argmax(output_contour, axis=3)
            output_mask = np.reshape(output_mask, (output_mask.shape[1], output_mask.shape[2]))
            output_contour = np.reshape(output_contour, (output_contour.shape[1], output_contour.shape[2]))

            return output_mask, output_contour
        else:
            raise NotImplemented("no model loaded")
