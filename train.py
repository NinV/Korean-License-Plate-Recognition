import argparse
import os
import tensorflow as tf
from tensorflow import keras
from model import LPRNet
from loader import Loader
from metrics import ctc_loss, Evaluator


def train(args):
    save_weights_only = args["save_weights_only"]
    loader = Loader(args["label"], args["img_dir"], load_all=args["load_all"])
    net = LPRNet(loader.get_num_chars() + 1)

    if args["pretrained"]:
        net.load_weights(args["pretrained"])
        print("Pretrained model loaded")

    model = net.model
    train_dataset = tf.data.Dataset.from_generator(loader,
                                                   output_types=(tf.float32, tf.int32, tf.int32)).batch(
        args["batch_size"]).shuffle(len(loader)).prefetch(tf.data.experimental.AUTOTUNE)
    print("Training data loaded")

    if args["valid_label"] and args["valid_img_dir"]:
        evaluator = Evaluator(net, args["valid_label"], args["valid_img_dir"], args["valid_batch_size"])
        print("Validation data loaded")
    else:
        evaluator = None

    learning_rate = keras.optimizers.schedules.ExponentialDecay(args["learning_rate"],
                                                                decay_steps=args["decay_steps"],
                                                                decay_rate=args["decay_rate"],
                                                                staircase=args["staircase"])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    best_val_loss = float("inf")
    for step, (imgs, labels, label_lengths) in enumerate(train_dataset):
        if step == args["num_steps"]:
            break
        with tf.GradientTape() as tape:
            logits = model(imgs, training=True)
            batch_size, times = logits.shape[:2]
            logits_lengths = tf.expand_dims(tf.tile(tf.constant([times], tf.int32),
                                                    tf.constant([batch_size], tf.int32)),
                                            axis=1)
            loss_value = ctc_loss(labels, logits, logits_lengths, label_lengths)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        loss_value = float(loss_value)
        print("[batch {} - Seen: {} samples] "
              "Training loss: {}, "
              "learning_rate: {} ".format(step + 1,
                                          (step + 1) * args["batch_size"],
                                          float(loss_value),
                                          optimizer._decayed_lr(
                                              tf.float32).numpy()
                                          ))

        # Log every 10 batches.
        if step % args["valid_interval"] == 0 and step > 0:
            if evaluator is not None:
                val_loss, _, _ = evaluator.evaluate()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_weights_only:
                        net.save_weights(os.path.join(args["saved_dir"], "weights_best.pb"))
                    else:
                        net.save(os.path.join(args["saved_dir"], "model_best.pb"))
                    print("save best at batch: {}, loss: {}".format(step + 1, val_loss))

    if save_weights_only:
        net.save_weights(os.path.join(args["saved_dir"], "weights_last.pb"))
    else:
        net.save(os.path.join(args["saved_dir"], "model_last.pb"))


def parser_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument("-l", "--label", required=True, help="Path to label file (training set)")
    parser.add_argument("-i", "--img_dir", required=True, help="Path to image folder (training set)")
    parser.add_argument("--valid_label", default="", help="Path to label file (validation set)")
    parser.add_argument("--valid_img_dir", default="", help="Path to image folder (validation set)")
    parser.add_argument("--valid_batch_size", type=int, default=850, help="Path to image folder (validation set)")
    parser.add_argument("--load_all", action="store_true", help="Load all dataset to RAM (for small dataset)")

    # save config
    parser.add_argument("-s", "--saved_dir", default="saved_models", help="folder for saving model")
    parser.add_argument("--save_weights_only", action="store_true", help="save weights only")
    parser.add_argument("--valid_interval", type=int, default=100, help="Save model and evaluate interval")

    # training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=10e-3, help="Initial learning rate")
    parser.add_argument("--decay_steps", type=float, default=10000, help="learning rate decay step")
    parser.add_argument("--decay_rate", type=float, default=0.995, help="learning rate decay rate")
    parser.add_argument("--staircase", action="store_true", help="learning rate decay on step (default: smooth)")

    # load pre-trained
    parser.add_argument("--pretrained", default="", help="Path to saved model")

    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    args = parser_args()
    tf.compat.v1.enable_eager_execution()
    train(args)
