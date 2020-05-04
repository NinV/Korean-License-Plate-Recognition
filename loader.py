import os
import json
import random
import cv2


def normalize(img):
    width = 94
    height = 24
    return cv2.resize(img, (width, height)) / 255


class Loader:
    def __init__(self, labelfile, img_dir, preproc_func=normalize):
        self.img_dir = img_dir
        self.preproc_func = preproc_func
        with open(labelfile, "r") as f:
            obj = json.load(f)
        self.data = obj["data"]
        self.class_names = obj["class_names"]
        self.lookup = {name: i for i, name in enumerate(self.class_names)}

    def get_num_chars(self):
        return len(self.class_names)

    def __iter__(self):
        random.shuffle(self.data)
        self.i = 0
        return self

    def __next__(self):
        if self.i >= len(self.data):
            self.i = 0
        img_data = self.data[self.i]
        img = cv2.imread(os.path.join(self.img_dir, img_data["images"]["filename"]))
        img = self.preproc_func(img)
        text = img_data["annotations"]["text"]
        label = [self.lookup[c] for c in text]
        self.i += 1
        return img, label, [len(label)]

    def __len__(self):
        return len(self.data)

    def __call__(self, *args, **kwargs):
        return self
