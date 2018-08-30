import tensorflow as tf
import os
import logging
import numpy as np
from net import adversarial_net
from data import segmentation_provider
from sklearn.utils import shuffle
from tensorflow.contrib import slim
from evaluator import Evaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class AdverTrainer(object):

    def __init__(self, net, batch_size=1, training_iters=10000, learning_rate=0.01):
        self.net = net
        self.batch_size = batch_size
        self.training_iters = training_iters
        discriminate_variables = net.get_discriminate_variable()
        generate_variables = net.get_generate_variable()
        self.learning_rate = tf.placeholder("float")
        self.judge_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            net.judge_loss,
            var_list=discriminate_variables)
        self.segmentation_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
            net.adversarial_loss,
            var_list=generate_variables)

    def adversarial_train(self, data_provider, output_path, training_epochs=100, display_step=1000, save_epoch=1,
                          verify_epoch=1,
                          start_adversarial=10, start_judge=5):
        save_path = os.path.join(output_path, "model.ckpt")
        tf.summary.scalar("adversarial_loss", self.net.adversarial_loss)
        tf.summary.scalar("judge_loss", self.net.judge_loss)
        dummy_mask = np.zeros((self.batch_size, self.net.height, self.net.width, self.net.n_class))
        merge_dummy_mask = np.zeros([self.batch_size * 2, self.net.height, self.net.width, self.net.n_class])
        data_provider.sample_test_data(data_provider.sample_size)
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            ckpt = tf.train.get_checkpoint_state(output_path)
            if ckpt and ckpt.model_checkpoint_path:
                variables = slim.get_variables_to_restore()
                variables_to_restore = [v for v in variables if v.name.split('/')[0] != "judgeNet"]
                saver = tf.train.Saver(variables_to_restore)
                saver.restore(sess, ckpt.model_checkpoint_path)
                logging.info("model restored from file " + ckpt.model_checkpoint_path)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            for epoch in range(training_epochs):
                for step in range(epoch * self.training_iters, (epoch + 1) * self.training_iters):
                    if epoch >= start_adversarial:
                        # train annotated data
                        batch_x, batch_y, batch_y_contour = data_provider(self.batch_size)
                        label = np.ones([self.batch_size, 1], dtype=np.float32)
                        _ = sess.run([self.segmentation_optimizer],
                                     feed_dict={self.net.x: batch_x,
                                                self.net.mask: batch_y,
                                                self.net.contour_mask: batch_y_contour,
                                                self.net.judge_label: label,
                                                self.net.judge_weight: 0.5,
                                                self.learning_rate: 1e-5})

                        # train test data
                        test_x = data_provider.get_test_data(self.batch_size)
                        test_label = np.zeros([self.batch_size, 1], dtype=np.float32)
                        _ = sess.run([self.segmentation_optimizer],
                                     feed_dict={
                                         self.net.x: test_x,
                                         self.net.mask: dummy_mask,
                                         self.net.contour_mask: dummy_mask,
                                         self.net.judge_label: test_label,
                                         self.net.judge_weight: 0.5,
                                         self.learning_rate: 1e-5
                                     })

                        # train discriminator
                        merge_data, merge_label = shuffle(np.concatenate([batch_x, test_x], axis=0),
                                             np.concatenate([label, test_label], axis=0))
                        _ = sess.run(self.judge_optimizer,
                                     feed_dict={
                                         self.net.x:merge_data,
                                         self.net.mask:merge_dummy_mask,
                                         self.net.contour_mask: merge_dummy_mask,
                                         self.net.judge_label: merge_label,
                                         self.net.judge_weight: 1.0,
                                         self.learning_rate: 1e-5
                                     })
                    elif epoch >= start_judge:
                        new_batch_size = 1
                        batch_x, batch_y, batch_y_contour = data_provider(new_batch_size)
                        test_x = data_provider.get_test_data(new_batch_size)
                        label = np.ones([new_batch_size, 1])
                        test_label = np.zeros([new_batch_size, 1])

                        merge_data, merge_label = shuffle(np.concatenate([batch_x, test_x], axis=0),
                                                          np.concatenate([label, test_label], axis=0))
                        new_dummy_mask = np.zeros(
                            [new_batch_size * 2, self.net.height, self.net.width, self.net.n_class])
                        _ = sess.run(self.judge_optimizer,
                                     feed_dict={
                                         self.net.x: merge_data,
                                         self.net.mask: new_dummy_mask,
                                         self.net.contour_mask: new_dummy_mask,
                                         self.net.judge_label: merge_label,
                                         self.net.judge_weight: 1.0,
                                         self.learning_rate: 1e-5
                                     })
                    else:
                        batch_x, batch_y, batch_y_contour = data_provider(self.batch_size)
                        label = np.ones([self.batch_size, 1])
                        _ = sess.run(self.segmentation_optimizer, feed_dict={self.net.x: batch_x,
                                                                             self.net.mask: batch_y,
                                                                             self.net.contour_mask: batch_y_contour,
                                                                             self.net.judge_label: label,
                                                                             self.net.judge_weight: 0.0,
                                                                             self.learning_rate: max(0.01 / (epoch + 1),
                                                                                                     1e-5)})

                    if step % display_step == 0:
                        summary_str, loss, judge_loss = sess.run(
                            [summary_op, self.net.adversarial_loss, self.net.judge_loss],
                            feed_dict={self.net.x: batch_x,
                                       self.net.mask: batch_y,
                                       self.net.contour_mask: batch_y_contour,
                                       self.net.judge_label: label,
                                       self.net.judge_weight: 1.0})
                        summary_writer.add_summary(summary_str, step)
                        summary_writer.flush()
                        logging.info("epoch {:}, step {:}, Minibatch Loss={:.4f}, {:.4f}".format(epoch,
                                                                                                 step,
                                                                                                 loss,
                                                                                                 judge_loss))

                        if epoch >= start_judge:
                            judge_loss = sess.run(
                                self.net.judge_loss,
                                feed_dict={self.net.x: test_x,
                                           self.net.mask: dummy_mask,
                                           self.net.contour_mask: dummy_mask,
                                           self.net.judge_label: test_label,
                                           self.net.judge_weight: 1.0})
                            logging.info("test epoch {:}, step {:}, Minibatch Loss={:.4f}".format(epoch,
                                                                                                  step,
                                                                                                  judge_loss))

                if epoch % save_epoch == 0:
                    self.net.save(sess, save_path + str(epoch))

                if epoch % verify_epoch == 0:
                    self.verification_evaluate_unet(self.net, sess, data_provider, epoch, summary_writer)

    @staticmethod
    def verification_evaluate_unet(net, sess, data_provider, epoch, summary_writer):
        test_x, test_y, test_y_contour = data_provider.verification_data()
        batch_size = len(test_x)
        if batch_size == 0:
            return
        total_loss = 0
        index = 0
        while index < batch_size:

            if index + 10 < batch_size:
                end = index + 10
            else:
                end = batch_size
            segmentation_loss = sess.run(
                net.segmentation_loss,
                feed_dict={net.x: np.asarray(test_x[index:end]),
                           net.mask: np.asarray(test_y[index:end]),
                           net.contour_mask: np.asarray(test_y_contour[
                                                        index:end])})
            total_loss += segmentation_loss
            index = end
        segmentation_loss = total_loss / batch_size * 10
        summary = tf.Summary()
        summary.value.add(tag="verification_loss", simple_value=segmentation_loss)
        logging.info("epoch: {:}, verification loss: {:.4f}".format(epoch, segmentation_loss))

        valid_data, valid_mask = data_provider.get_valid_data()
        average_f1 = 0.0
        average_aji = 0.0
        average_dice = 0.0
        for i in range(len(valid_data)):
            mask = predict(valid_data[i], net, sess)
            f1_score, _, _ = Evaluator.f1_score(valid_mask[i].astype(np.int32), mask.astype(np.int32))
            aji = Evaluator.aji(valid_mask[i].astype(np.int32), mask.astype(np.int32))
            dice = Evaluator.dice(valid_mask[i].astype(np.int32), mask.astype(np.int32))
            average_f1 += f1_score
            average_aji += aji
            average_dice += dice
        average_f1 /= len(valid_data)
        average_aji /= len(valid_data)
        average_dice /= len(valid_data)
        print("average f1 " + str(average_f1))
        print("average aji " + str(average_aji))
        print("average dice " + str(average_dice))
        summary.value.add(tag="f1", simple_value=average_f1)
        summary.value.add(tag="aji", simple_value=average_aji)
        summary.value.add(tag="dice", simple_value=average_dice)
        summary_writer.add_summary(summary, epoch)
        summary_writer.flush()


def predict(test_image, net, sess, sample_size=225, output_size=225):
    """
    given an image, predict segmentation mask
    :param test_image: image array
    :return: mask array
    """
    test_image = (test_image - np.average(test_image)) / np.std(test_image)

    height, width = test_image.shape[0], test_image.shape[1]
    mask = np.zeros((height, width))
    contour = np.zeros((height, width))
    test_image = np.pad(test_image, ((sample_size, sample_size), (sample_size, sample_size),
                                     (0, 0)), "reflect")
    x, y = 0, 0
    stride = int((sample_size - output_size) / 2)
    while x < height or y < width:
        sample = test_image[x + sample_size - stride: x + sample_size + output_size + stride,
                 y + sample_size - stride: y + sample_size + output_size + stride, 0:3]
        x_max = min(x + output_size, height)
        y_max = min(y + output_size, width)
        sample_list = [sample, (np.rot90(sample, 2)), (np.flip(sample, axis=0)), (np.flip(sample, axis=1))]
        sample_mask_list, sample_contour_list = net.predict(np.asarray(sample_list), sess=sess)
        sample_mask = np.average(np.asarray([sample_mask_list[0], np.rot90(sample_mask_list[1], 2),
                                             np.flip(sample_mask_list[2], axis=0),
                                             np.flip(sample_mask_list[3], axis=1)]), axis=0)
        sample_contour = np.average(np.asarray([sample_contour_list[0], np.rot90(sample_contour_list[1], 2),
                                                np.flip(sample_contour_list[2], axis=0),
                                                np.flip(sample_contour_list[3], axis=1)]), axis=0)
        mask[x:x_max, y:y_max] = sample_mask[:x_max - x, :y_max - y]
        contour[x:x_max, y:y_max] = sample_contour[:x_max - x, :y_max - y]

        # update index
        if x + output_size < height:
            x += output_size
        elif y + output_size < width:
            y += output_size
            x = 0
        else:
            break
    return Evaluator.post_process(mask, contour)


if __name__ == "__main__":
    net = adversarial_net.AdversarialNet()
    data_provider = segmentation_provider.SegmentationDataProvider("/data/Cell/norm_data/training_data/",
                                                                   "/data/Cell/norm_data/test_data/", resample=True,
                                                                   sample_number=3000, sample_size=225, output_size=225,
                                                                   train_percent=0.9)
    trainer = AdverTrainer(net, batch_size=1, learning_rate=0.001, training_iters=5000)
    trainer.adversarial_train(data_provider, "/data/Cell/yunzhe/resnet_attention/", start_adversarial=30, start_judge=30,
                              display_step=500, training_epochs=30)
