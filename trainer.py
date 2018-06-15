import tensorflow as tf
import os
import logging
import numpy as np
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Trainer(object):

    def __init__(self, net, batch_size=1, optimizer="momentum", learning_rate=0.01, decay_rate=0.95,
                 momentum=0.9, decay_step=32):
        """
        initializer
        :param net: the network graph
        :param batch_size: training batch_size
        :param optimizer: adam or momentum
        :param learning_rate: learning rate
        :param decay_rate: decay rate
        :param momentum: momentum
        """
        # store all hyper parameters
        self.net = net
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.momentum = momentum
        self.global_step = tf.Variable(0)
        self.decay_step = decay_step

        if optimizer == "momentum":
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=decay_step,
                                                            decay_rate=decay_rate,
                                                            staircase=True)
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node,
                                                        momentum=self.momentum).minimize(self.net.cost,
                                                                                         global_step=self.global_step)
        elif optimizer == "adam":
            learning_rate_node = tf.train.exponential_decay(learning_rate=learning_rate,
                                                            global_step=self.global_step,
                                                            decay_steps=decay_step,
                                                            decay_rate=decay_rate,
                                                            staircase=True)
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate_node).minimize(self.net.cost,
                                                                                                   global_step=self.global_step)

        else:
            raise NameError("unknown optimizer name")

    def train(self, data_provider, output_path, training_iters=32, epochs=10,
              display_step=10, save_epoch=5, restore=False, verify_epoch=10):
        save_path = os.path.join(output_path, "model.ckpt")
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver = tf.train.Saver()
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    logging.info("model restored from file " + ckpt.model_checkpoint_path)

            for epoch in range(epochs):

                if epoch % save_epoch == 0:
                    self.net.save(sess, save_path + str(epoch))

                total_loss = 0
                for step in range(training_iters):
                    batch_x, batch_y, batch_y_contour = data_provider(self.batch_size)

                    _, loss = sess.run([self.optimizer, self.net.cost],
                                       feed_dict={self.net.x: batch_x,
                                                  self.net.mask: batch_y,
                                                  self.net.contour_mask: batch_y_contour
                                       })
                    total_loss += loss

                    if (epoch * training_iters + step) % display_step == 0:
                        logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                                         step,
                                                                                         loss))

                if epoch % verify_epoch == 0:
                    self.verification_evaluate(sess, data_provider, epoch)

        return save_path

    def verification_evaluate(self, sess, data_provider, epoch):
        test_x, test_y, test_y_contour = data_provider.verification_data()
        batch_size = len(test_x)
        total_loss = 0
        index = 0
        while index < batch_size:

            if index + 50 < batch_size:
                end = index + 50
            else:
                end = batch_size
            verification_loss = sess.run(self.net.cost, feed_dict={self.net.x: np.asarray(test_x)[index:end, ...],
                                                                   self.net.mask: np.asarray(test_y)[index:end, ...],
                                                                   self.net.contour_mask: np.asarray(test_y_contour)[index:end, ...]})
            total_loss += verification_loss
            index = end

        logging.info("epoch: {:}, verification loss: {:.4f}".format(epoch,
                                                                    total_loss/batch_size * 50))
