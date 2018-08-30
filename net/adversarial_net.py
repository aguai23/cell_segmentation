import tensorflow as tf
from tensorflow.contrib import slim
from resnet import resnet_v1
import logging
import numpy as np


def bce(o, t):
    o = tf.clip_by_value(o, 1e-7, 1. - 1e-7)
    return -(t * tf.log(o) + (1. - t) * tf.log(1. - o))


class AdversarialNet(object):

    def __init__(self, output_size=225, height=None, width=None):
        self.n_class = 2
        self.height = height
        self.width = width
        if not height:
            self.height = output_size
        if not width:
            self.width = output_size
        tf.reset_default_graph()
        self.x = tf.placeholder("float", shape=[None, self.height, self.width, 3])
        self.mask = tf.placeholder("float", shape=[None, self.height, self.width, 2])
        self.contour_mask = tf.placeholder("float", shape=[None, self.height, self.width, 2])
        self.output_mask, self.output_contour, self.mask_feature = self.resnet(self.x)

        self.segmentation_loss = self.segmentation_cost(self.output_mask, self.output_contour)

        self.judge_label = tf.placeholder("float", shape=[None, 1])
        self.judge_weight = tf.placeholder("float")
        self.output_label = self.judgenet(self.output_mask[..., 1:], self.output_contour[..., 1:], self.x)

        self.judge_loss = self.judge_cost(self.output_label, self.judge_label)
        self.adversarial_loss = tf.reduce_mean(
            self.judge_label * self.segmentation_loss + self.judge_weight * (1. - self.judge_label) *
            bce(self.output_label, (1. - self.judge_label)))
        self.predict_mask = tf.nn.softmax(self.output_mask)
        self.predict_contour = tf.nn.softmax(self.output_contour)
        self.session = None

    @staticmethod
    def judge_cost(output_label, judge_label):
        regularization = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope="judgeNet")
        cost = bce(output_label, judge_label)
        return tf.reduce_mean(cost) + tf.reduce_mean(regularization)

    def auxilary_loss(self, mask_feature):
        output_mask = slim.conv2d(mask_feature, 2, 1, activation_fn=None, normalizer_fn=None)
        new_height = int((self.height + 3) / 4)
        new_width = int((self.width + 3) / 4)
        resize_mask = tf.image.resize_nearest_neighbor(self.mask, size=[new_height, new_width])
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(output_mask, [-1, self.n_class]),
                                                                      labels=tf.reshape(resize_mask, [-1, self.n_class])))

        return cost

    def segmentation_cost(self, output_map, output_contour, contour_weight=2.0, nuclei_weight=1.0,
                          background_weight=2.0, gamma=0):
        flat_output_map = tf.reshape(output_map, [-1, self.n_class])
        output_map_prob = tf.nn.softmax(flat_output_map)
        flat_mask = tf.reshape(self.mask, [-1, self.n_class])
        focal_weight = nuclei_weight * flat_mask[..., 1] + \
                       background_weight * flat_mask[..., 0]
        cross_entropy_cost = tf.reduce_mean(tf.multiply(tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_map,
                                                                                                labels=flat_mask),
                                                        focal_weight))
        regularization = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        flat_output_contour = tf.reshape(output_contour, [-1, self.n_class])
        flat_contour_mask = tf.reshape(self.contour_mask, [-1, self.n_class])
        output_contour_prob = tf.nn.softmax(flat_output_contour)
        cross_entropy_contour = tf.nn.softmax_cross_entropy_with_logits(logits=flat_output_contour,
                                                                        labels=flat_contour_mask)
        weight_map = contour_weight * flat_contour_mask[..., 1] + flat_contour_mask[..., 0]
        weight_contour_loss = tf.reduce_mean(tf.multiply(weight_map, cross_entropy_contour))

        cost = cross_entropy_cost + weight_contour_loss + regularization
        return cost

    def resnet(self, image):
        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }
        with slim.arg_scope([slim.conv2d],
                            weights_regularizer=slim.l2_regularizer(1e-4),
                            weights_initializer=slim.variance_scaling_initializer(),
                            activation_fn=tf.nn.relu,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                with slim.arg_scope([slim.max_pool2d], padding="SAME"):
                    logits, end_points = resnet_v1.resnet_v1_50(image, is_training=True, scope="resnet_v1_50")

        with tf.variable_scope('feature_fusion', values=[end_points.values]):

            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(1e-4)):
                f = [end_points['pool5'], end_points['pool4'],
                     end_points['pool3'], end_points['pool2']]
                global_pool = tf.reduce_mean(f[0], [1, 2], name='pool5', keep_dims=True)
                for i in range(4):
                    print('Shape of f_{} {}'.format(i, f[i].shape))
                g = [None, None, None, None]
                h = [None, None, None, None]
                num_outputs = [256, 128, 64, 32, 32, 16]
                for i in range(4):
                    if i == 0:
                        global_pool = self.unpool(global_pool, size=[tf.shape(f[i])[1], tf.shape(f[i])[2]])
                        global_pool = self.residual_block(global_pool, num_outputs[i])
                        # h[i] = tf.concat([f[i], global_pool],
                        #                  axis=-1)
                        f[i] = self.residual_block(f[i], num_outputs[i])
                        h[i] = self.channel_block(global_pool, f[i], num_outputs[i])
                    else:
                        # c1_1 = tf.concat([g[i - 1], f[i]], axis=-1)
                        f[i] = self.residual_block(f[i], num_outputs[i])
                        h[i] = self.channel_block(g[i - 1], f[i], num_outputs[i])

                    if i < 3:
                        g[i] = self.residual_block(h[i], num_outputs[i])
                        g[i] = self.unpool(g[i], size=[tf.shape(f[i + 1])[1], tf.shape(f[i + 1])[2]])

                    if i == 3:
                        # c1_1 = slim.conv2d(tf.concat([g[i], end_points['pool1']], axis=-1), num_outputs[i], 1)
                        g[i] = self.unpool(h[i])
                        g[i] = self.residual_block(g[i], num_outputs[i])
                    print('Shape of h_{} {}, g_{} {}'.format(i, h[i].shape, i, g[i].shape))

                mask = self.unpool(g[3])
                mask = self.residual_block(mask, num_outputs[-1])
                output_map = slim.conv2d(mask, 2, 1, activation_fn=None, normalizer_fn=None)
                output_contour = slim.conv2d(mask, 2, 1, activation_fn=None, normalizer_fn=None)

        return output_map, output_contour, h[3]

    @staticmethod
    def channel_block(high_stage, low_stage, output_channel):
        attention_vector = tf.reduce_mean(high_stage, [1, 2], keep_dims=True)
        attention_vector = slim.conv2d(attention_vector, output_channel/16, [1, 1], activation_fn=tf.nn.relu)
        attention_vector = slim.conv2d(attention_vector, output_channel, [1, 1], activation_fn=tf.nn.sigmoid)
        high_stage = slim.conv2d(high_stage, output_channel, [1, 1])
        output = low_stage * attention_vector + high_stage
        return output

    @staticmethod
    def residual_block(input, output_channel):
        plain = slim.conv2d(input, output_channel, [1, 1])
        conv = slim.conv2d(plain, output_channel, [3, 3], activation_fn=None)
        conv = slim.batch_norm(conv)
        conv = tf.nn.relu(conv)
        conv = slim.conv2d(conv, output_channel, [3, 3], activation_fn=None)
        output = tf.nn.relu(conv + plain)
        return output

    @staticmethod
    def unpool(inputs, size=None):
        if size:
            return tf.image.resize_bilinear(inputs, size=size)
        else:
            return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1] * 2 - 1, tf.shape(inputs)[2] * 2 - 1])

    def judgenet(self, mask, contour, image):
        ensamble_mask = tf.multiply(mask, image)
        reverse_mask = tf.multiply(1 - mask, image)
        ensamble_contour = tf.multiply(contour, image)
        reverse_contour = tf.multiply(1 - contour, image)
        input = tf.concat([ensamble_mask, reverse_mask, ensamble_contour, reverse_contour], axis=-1)

        with tf.variable_scope('judgeNet', values=[input]):
            batch_norm_params = {
                'decay': 0.997,
                'epsilon': 1e-5,
                'scale': True,
                'is_training': True
            }
            with slim.arg_scope([slim.conv2d],
                                activation_fn=tf.nn.leaky_relu,
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                weights_regularizer=slim.l2_regularizer(1e-4),
                                padding="SAME"):
                with slim.arg_scope([slim.max_pool2d], padding="SAME"):
                    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.leaky_relu,
                                        weights_regularizer=slim.l2_regularizer(1e-4)):
                        conv1 = self.residual_block(input, 32)
                        pool1 = slim.avg_pool2d(conv1, [2, 2], stride=2)

                        conv2 = self.residual_block(pool1, 64)
                        pool2 = slim.avg_pool2d(conv2, [2, 2], stride=2)

                        conv3 = self.residual_block(pool2, 128)

                        global_pool = tf.reduce_mean(conv3, [1, 2], keep_dims=False)
                        output = slim.fully_connected(global_pool, 1, activation_fn=None)
                        output = tf.nn.sigmoid(output)

        return output

    @staticmethod
    def get_generate_variable():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="resnet_v1") + \
                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="feature_fusion")
        return variables

    @staticmethod
    def get_discriminate_variable():
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="judgeNet")
        return variables

    @staticmethod
    def save(sess, model_path):
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

    def predict(self, x_test, mask=None, contour=None, sess=None):
        if sess is None and self.session:
            sess = self.session

        mask_dummy = np.empty([x_test.shape[0], self.height, self.width, self.n_class])
        contour_dummy = np.empty([x_test.shape[0], self.height, self.width, self.n_class])
        label_dummy = np.empty([x_test.shape[0], 1])
        if mask is not None and contour is not None:
            cost, output_mask, output_contour = sess.run(
                [self.segmentation_loss, self.predict_mask, self.predict_contour],
                feed_dict={self.x: x_test,
                           self.mask: mask,
                           self.contour_mask: contour})
            print(cost)
        else:
            output_mask, output_contour = sess.run([self.predict_mask, self.predict_contour],
                                                   feed_dict={self.x: x_test,
                                                              self.mask: mask_dummy,
                                                              self.contour_mask: contour_dummy,
                                                              self.judge_label: label_dummy,
                                                              self.judge_weight: 0.0})
        return output_mask[..., 1], output_contour[..., 1]

