import tensorflow as tf
from net.ops import conv2d
from tensorflow.contrib.layers import conv2d_transpose
from tensorflow.contrib import slim
from resnet import resnet_v1
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class DeepContourNet(object):

    def __init__(self, n_class=2, cost="cross_entropy", sample_size=129, output_size=123, height=None, width=None):
        self.height = height
        self.width = width
        if not height:
            self.height = output_size
        if not width:
            self.width = output_size
        tf.reset_default_graph()
        self.n_class = n_class
        self.x = tf.placeholder("float", shape=[None, self.height, self.width, 3])
        self.mask = tf.placeholder("float", shape=[None, self.height, self.width, 2])
        self.contour_mask = tf.placeholder("float", shape=[None, self.height, self.width, 2])
        self.output_mask, self.output_contour = self.resnet(self.x)
        self.cost = self.get_cost(self.output_mask, self.output_contour, cost, nuclei_weight=2, contour_weight=1 ,
                                  background_weight=1)
        self.cross_entropy_nuclei = self.get_cost(self.output_mask, self.output_contour, "dice", nuclei_weight=0)
        self.cross_entropy_contour = self.get_cost(self.output_mask, self.output_contour, "cross_entropy_contour",
                                                   contour_weight=4)
        self.predict_mask = tf.nn.softmax(self.output_mask)
        self.predict_contour = tf.nn.softmax(self.output_contour)
        self.session = None

    def build_net(self, input, scope=""):

        with tf.name_scope(scope, "dcnn"):
            # down sample
            conv0 = conv2d(input, 32, [3, 3], scope="conv0", padding="SAME", batch_norm_params={}, weight_decay=0.1)
            maxpool0 = tf.nn.max_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                      name="maxpool0")

            conv1 = conv2d(maxpool0, 64, [3, 3], scope="conv1", padding="SAME", batch_norm_params={}, weight_decay=0.1)
            maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME",
                                      name="maxpool1")

            conv2 = conv2d(maxpool1, 128, [3, 3], scope="conv2", padding="SAME", batch_norm_params={}, weight_decay=0.1)

            # up sample
            deconv0 = conv2d_transpose(conv2, 64, [2, 2], [2, 2], padding="SAME", activation_fn=tf.nn.relu,
                                       weights_regularizer=tf.nn.l2_loss)
            deconv0_concat = tf.concat([deconv0, conv1], 3)
            deconv0_out = conv2d(deconv0_concat, 64, [3, 3], scope="deconv0_out", padding="SAME", batch_norm_params={},
                                 weight_decay=0.1)

            deconv1 = conv2d_transpose(deconv0_out, 32, [2, 2], [2, 2], padding="SAME", activation_fn=tf.nn.relu,
                                       weights_regularizer=tf.nn.l2_loss)
            deconv1_concat = tf.concat([deconv1, conv0], 3)
            deconv1_out = conv2d(deconv1_concat, 32, [3, 3], scope="deconv1_out", padding="SAME", batch_norm_params={},
                                 weight_decay=0.1)

            output_mask = conv2d(deconv1_out, 1, [3, 3], scope="output_mask", padding="SAME", activation=tf.nn.sigmoid)

            return output_mask

    def resnet(self, image):
        with slim.arg_scope(resnet_v1.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_50(image, is_training=True, scope="resnet_v1_50")

        with tf.variable_scope('feature_fusion', values=[end_points.values]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': True
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(1e-4)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2'],
                     end_points['pool1'], end_points['pool0']]
                global_pool = tf.reduce_mean(f[0], [1, 2], name='pool5', keep_dims=True)
                for i in range(6):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None, None, None]
                h = [None, None, None, None, None, None]
                num_outputs = [128, 128, 64, 32, 32, 16]
                for i in range(4):
                    if i == 0:
                        global_pool = self.unpool(global_pool, size=[tf.shape(f[i])[1], tf.shape(f[i])[2]])
                        h[i] = tf.concat([f[i], global_pool],
                                         axis=-1)
                        h[i] = slim.conv2d(h[i], num_outputs[i], 3)
                        # h[i] = f[i]
                    else:
                        c1_1 = slim.conv2d(tf.concat([g[i - 1], f[i]], axis=-1), num_outputs[i], 1)
                        h[i] = slim.conv2d(c1_1, num_outputs[i], [3, 3])

                    if i < 3:
                        g[i] = self.unpool(h[i], size=[tf.shape(f[i + 1])[1], tf.shape(f[i + 1])[2]])
                    if i == 3:
                        # c1_1 = slim.conv2d(tf.concat([g[i], end_points['pool1']], axis=-1), num_outputs[i], 1)
                        g[i] = self.unpool(h[i])
                        g[i] = slim.conv2d(g[i], num_outputs[i], 3, padding="SAME")
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

                mask = self.unpool(g[3])
                mask = slim.conv2d(mask, num_outputs[3], 3, padding="SAME")
                output_map = slim.conv2d(mask, 2, 1, activation_fn=None, normalizer_fn=None)
                output_contour = slim.conv2d(mask, 2, 1, activation_fn=None, normalizer_fn=None)

        return output_map, output_contour

    @staticmethod
    def unpool(inputs, size=None):
        if size:
            return tf.image.resize_bilinear(inputs, size=size)
        else:
            return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2, tf.shape(inputs)[2] * 2])

    def get_cost(self, output_map, output_contour, cost_name, contour_weight=1.0, nuclei_weight=1.0,
                 background_weight=1.0):
        flat_output_map = tf.reshape(output_map, [-1, self.n_class])
        flat_mask = tf.reshape(self.mask, [-1, self.n_class])
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        # contour loss
        flat_output_contour = tf.reshape(output_contour, [-1, self.n_class])
        flat_contour_mask = tf.reshape(self.contour_mask, [-1, self.n_class])
        cross_entropy_contour = tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_contour,
                                                                        labels=flat_contour_mask)
        weight_map = contour_weight * flat_mask[..., 1] + 1
        weight_contour_loss = tf.reduce_mean(tf.multiply(weight_map, cross_entropy_contour))

        # cross entropy loss
        thresh_mask = nuclei_weight * flat_mask[..., 1] + background_weight * flat_mask[..., 0]
        cross_entropy_cost = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_map,
                                                                                                labels=flat_mask),
                                                        thresh_mask))
        # dice coefficient
        prediction = tf.nn.softmax(output_map)[..., 1:]
        mask = self.mask[..., 1:]
        intersection = tf.reduce_sum(prediction * mask)
        union = 1e-5 + tf.reduce_sum(prediction) + tf.reduce_sum(mask)
        dice_cost = 1 - (2 * intersection / union)

        # square loss
        mask = self.mask[..., 1:]
        flat_mask = tf.reshape(mask, [-1, 1])
        flat_logits = tf.nn.sigmoid(tf.reshape(output_map, [-1, 1]))
        square_error = tf.square(tf.subtract(flat_mask, flat_logits))
        thresh_mask = tf.abs(tf.subtract(flat_mask, flat_logits))
        square_error = tf.multiply(square_error, thresh_mask)
        square_cost = tf.reduce_mean(square_error)

        if cost_name == "cross_entropy":
            cost = cross_entropy_cost
        elif cost_name == "dice":
            cost = dice_cost
        elif cost_name == "square":
            cost = square_cost
        elif cost_name == "fuse":
            cost = weight_contour_loss + cross_entropy_cost + tf.reduce_mean(regularization)
        elif cost_name == "cross_entropy_contour":
            cost = weight_contour_loss
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
            mask_dummy = np.empty([x_test.shape[0], self.height, self.width, self.n_class])
            contour_dummy = np.empty([x_test.shape[0], self.height, self.width, self.n_class])
            if mask is not None and contour is not None:
                cost, output_mask, output_contour = sess.run([self.cost, self.predict_mask, self.predict_contour],
                                             feed_dict={self.x: x_test,
                                                        self.mask: mask,
                                                        self.contour_mask: contour})
                print(cost)
            else:
                output_mask, output_contour, _ = sess.run([self.predict_mask, self.predict_contour, self.cost],
                                                          feed_dict={self.x: x_test,
                                                                     self.mask: mask_dummy,
                                                                     self.contour_mask: contour_dummy})
            # print(output_mask)

            # output_mask = np.divide(np.exp(output_mask), np.sum(np.exp(output_mask), axis=3, keepdims=True))[..., 1]
            # output_mask = np.argmax(output_mask, axis=3)
            # output_mask = np.reshape(output_mask, (output_mask.shape[1], output_mask.shape[2]))
            # output_contour = np.divide(np.exp(output_contour), np.sum(np.exp(output_contour), axis=3, keepdims=True))[..., 1]
            # output_contour = np.reshape(output_contour, [output_contour.shape[1], output_contour.shape[2]])
            # output_mask = 1 / (1 + np.exp(-output_mask))
            # output_contour = 1 / (1 + np.exp(-output_contour))

            return output_mask[..., 1], output_contour[..., 1]
        else:
            raise NotImplemented("no model loaded")
