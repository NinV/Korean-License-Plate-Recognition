import tensorflow as tf
import argparse
from metrics import Evaluator
from model import LPRNet

classnames = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "가", "나", "다", "라", "마", "거", "너", "더", "러",
              "머", "버", "서", "어", "저", "고", "노", "도", "로",
              "모", "보", "소", "오", "조", "구", "누", "두", "루",
              "무", "부", "수", "우", "주", "허", "하", "호"
              ]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--label", required=True, help="Path to label file (training set)")
    parser.add_argument("-i", "--img_dir", required=True, help="Path to image folder (training set)")
    parser.add_argument("-w", "--weights", required=True, help="path to weights file")
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    args = vars(parser.parse_args())

    tf.compat.v1.enable_eager_execution()
    net = LPRNet(len(classnames) + 1)
    net.load_weights(args["weights"])

    evaluator = Evaluator(net, args["label"], args["img_dir"], args["batch_size"])
    evaluator.evaluate()
