import random
import json
import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from model import LPRNet
from loader import Loader


def ctc_loss(labels, predicts, input_lengths, label_lengths):
    loss = tf.keras.backend.ctc_batch_cost(
        labels,
        predicts,
        input_lengths,
        label_lengths)
    loss = tf.keras.backend.mean(loss)
    return loss


def train(model, train_dataset, epochs):
    optimizer = keras.optimizers.SGD(learning_rate=1e-3)
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        for step, (imgs, labels, label_lengths) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(imgs, training=True)  # Logits for this minibatch
                batch_size, times = logits.shape[:2]
                logits_lengths = tf.expand_dims(tf.tile(tf.constant([times], tf.int32),
                                                        tf.constant([batch_size], tf.int32)),
                                                axis=1)
                # Compute the loss value for this minibatch.
                loss_value = ctc_loss(labels, logits, logits_lengths, label_lengths)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 200 batches.
            if step % 10 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()

    batch_size = 32
    labelfile = "/home/ninv/myProjects/dataset/KarPlate_Dataset/dataConverter/label.json"
    img_dir = "/home/ninv/myProjects/dataset/KarPlate_Dataset/dataConverter/Subset_LPR/train/images"
    loader = Loader(labelfile, img_dir)
    dataset = tf.data.Dataset.from_generator(loader, output_types=(tf.float32, tf.int32, tf.int32)).batch(batch_size)

    lprnet = LPRNet(loader.get_num_chars() + 1)
    train(lprnet.model, dataset, epochs=1)
