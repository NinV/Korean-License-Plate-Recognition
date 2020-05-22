import tensorflow as tf
import argparse
from metrics import Evaluator
from model import LPRNet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", required=True, help="Path to label file (training set)")
    parser.add_argument("-i", "--img_dir", required=True, help="Path to image folder (training set)")
    parser.add_argument("-w", "--weights", required=True, help="path to weights file")
    parser.add_argument("-b", "--batch_size", type=int, default=1000)
    args = vars(parser.parse_args())

    evaluator = Evaluator(None, args["label"], args["img_dir"], args["batch_size"])
    tf.compat.v1.enable_eager_execution()
    net = LPRNet(len(evaluator.loader.class_names) + 1)
    net.load_weights(args["weights"])
    evaluator.model = net
    evaluator.evaluate()
