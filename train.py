import argparse
import os
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


def train(args):
    loader = Loader(args["label"], args["img_dir"])
    net = LPRNet(loader.get_num_chars() + 1)
    model = net.model
    train_dataset = tf.data.Dataset.from_generator(loader,
                                                   output_types=(tf.float32, tf.int32, tf.int32)).batch(args["batch_size"]).repeat(1)

    optimizer = keras.optimizers.Adam(learning_rate=args["learning_rate"])
    best_loss = float("inf")
    for batch, (imgs, labels, label_lengths) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)  # Logits for this minibatch
            batch_size, times = logits.shape[:2]
            logits_lengths = tf.expand_dims(tf.tile(tf.constant([times], tf.int32),
                                                    tf.constant([batch_size], tf.int32)),
                                            axis=1)
            loss_value = ctc_loss(labels, logits, logits_lengths, label_lengths)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_value = float(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            net.save(os.path.join(args["saved_dir"], "model_best.pb"))
            print("save best at batch: {}, loss: {}".format(batch + 1, loss_value))

        # Log every 10 batches.
        if batch % 10 == 0:
            print("[batch {}]Training loss: {}. Seen: {} samples".format(batch + 1,
                                                                         float(loss_value),
                                                                         (batch + 1) * args["batch_size"]))

        if batch == args["num_steps"]:
            break
    net.save(os.path.join(args["saved_dir"], "model_last.pb"))


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", required=True, help="Path to label file")
    parser.add_argument("-i", "--img_dir", required=True, help="Path to image folder")
    parser.add_argument("-s", "--saved_dir", default="saved_models", help="folder for saving model")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=10e-4)
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained model")
    parser.add_argument("--saved_model", help="Path to saved_model")
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parser_args()
    tf.compat.v1.enable_eager_execution()
    train(args)
