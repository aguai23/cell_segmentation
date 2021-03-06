import tensorflow as tf
import os
import logging
import numpy as np
import queue

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
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node).minimize(self.net.cost,
                                                                                               global_step=self.global_step)

        else:
            raise NameError("unknown optimizer name")

        tf.summary.scalar("loss", self.net.cost)
        self.summary_op = tf.summary.merge_all()

    def train_unet(self, data_provider, output_path, training_iters=32, epochs=10,
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

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            training_queue = queue.Queue()
            previous_queue = queue.Queue()
            for epoch in range(epochs):

                total_loss = 0
                thresh = max(0.2, 1 - epoch / 10)
                for step in range(epoch * training_iters, (epoch + 1) * training_iters):
                    if not previous_queue.empty():
                        hard_sample = previous_queue.get()
                        batch_x, batch_y, batch_y_contour = hard_sample[0], hard_sample[1], hard_sample[2]
                    else:
                        batch_x, batch_y, batch_y_contour = data_provider(self.batch_size)

                    if (epoch * training_iters + step) % display_step == 0:
                        summary_str, loss, output_mask = sess.run(
                            [self.summary_op, self.net.cost, self.net.output_mask],
                            feed_dict={self.net.x: batch_x,
                                       self.net.mask: batch_y,
                                       self.net.contour_mask: batch_y_contour})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                                         step,
                                                                                         loss))

                    _, loss = sess.run([self.optimizer, self.net.cross_entropy_nuclei],
                                       feed_dict={self.net.x: batch_x,
                                                  self.net.mask: batch_y,
                                                  self.net.contour_mask: batch_y_contour
                                                  })
                    if loss > thresh:
                        training_queue.put([batch_x, batch_y, batch_y_contour])
                while not training_queue.empty():
                    previous_queue.put(training_queue.get())
                if epoch % verify_epoch == 0:
                    logging.info("hard sample number " + str(previous_queue.qsize()))
                    self.verification_evaluate_unet(sess, data_provider, epoch, summary_writer)

                if epoch % save_epoch == 0:
                    self.net.save(sess, save_path + str(epoch))

        return save_path

    def train_classification(self, data_provider, output_path, training_iters=32, epochs=10,
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

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            for epoch in range(epochs):

                if epoch % save_epoch == 0:
                    self.net.save(sess, save_path + str(epoch))

                total_loss = 0
                for step in range(epoch * training_iters, (epoch + 1) * training_iters):
                    batch_x, batch_y = data_provider(self.batch_size)

                    if (epoch * training_iters + step) % display_step == 0:
                        summary_str, loss, logits = sess.run([self.summary_op, self.net.cost, self.net.logits],
                                                             feed_dict={self.net.x: batch_x,
                                                                        self.net.y: batch_y})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                                         step,
                                                                                         loss))
                        # print(logits)

                    _, _ = sess.run([self.optimizer, self.net.cost],
                                    feed_dict={self.net.x: batch_x,
                                               self.net.y: batch_y})

                if epoch % verify_epoch == 0:
                    self.verification_evaluate_classification(sess, data_provider, epoch, summary_writer)

        return save_path

    def verification_evaluate_unet(self, sess, data_provider, epoch, summary_writer):
        test_x, test_y, test_y_contour = data_provider.verification_data()
        batch_size = len(test_x)
        if batch_size == 0:
            return
        total_loss_nuclei = 0
        total_loss_contour = 0
        index = 0
        total_contour_accuracy = 0.0
        while index < batch_size:

            if index + 10 < batch_size:
                end = index + 10
            else:
                end = batch_size
            nuclei_loss, contour_loss, predict_contour = sess.run(
                [self.net.cross_entropy_nuclei, self.net.cross_entropy_contour,
                 self.net.predict_contour],
                feed_dict={self.net.x: np.asarray(test_x[index:end]),
                           self.net.mask: np.asarray(test_y[index:end]),
                           self.net.contour_mask: np.asarray(test_y_contour[
                                                             index:end])})
            predict_contour[np.where(predict_contour > 0.5)] = 1
            predict_contour[np.where(predict_contour <= 0.5)] = 0
            contour_accuracy = np.sum(np.multiply(predict_contour,
                                                  test_y_contour[index:end])) / np.sum(test_y_contour[index:end])

            total_loss_nuclei += nuclei_loss
            total_loss_contour += contour_loss
            total_contour_accuracy += contour_accuracy
            index = end
        nuclei_loss = total_loss_nuclei / batch_size * 10
        contour_loss = total_loss_contour / batch_size * 10
        contour_accuracy = total_contour_accuracy / batch_size * 10
        summary = tf.Summary()
        summary.value.add(tag="nuclei_loss", simple_value=nuclei_loss)
        summary.value.add(tag="contour_loss", simple_value=contour_loss)
        summary.value.add(tag="contour_accuracy", simple_value=contour_accuracy)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
        logging.info("epoch: {:}, verification loss: {:.4f} {:.4f} {:.4f}".format(epoch,
                                                                           nuclei_loss, contour_loss, contour_accuracy))

    def verification_evaluate_classification(self, sess, data_provider, epoch, summary_writer):
        test_x, test_y = data_provider.verification_data()
        batch_size = len(test_x)
        total_loss = 0
        index = 0
        while index < batch_size:

            if index + 100 < batch_size:
                end = index + 100
            else:
                end = batch_size
            verification_loss = sess.run(self.net.cost,
                                         feed_dict={self.net.x: np.asarray(test_x)[index:end, ...],
                                                    self.net.y: np.asarray(test_y)[index:end, ...]})

            total_loss += verification_loss
            index = end
        epoch_loss = total_loss / batch_size * 100
        summary = tf.Summary()
        summary.value.add(tag="verification_loss", simple_value=epoch_loss)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()
        logging.info("epoch: {:}, verification loss: {:.4f}".format(epoch,
                                                                    epoch_loss))
