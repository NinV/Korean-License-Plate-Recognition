import random
import numpy as np
import cv2


class IntensityDistortion:
    def random_distort(self, image, mode="bgr"):
        raise NotImplemented


class ColorDistorion(IntensityDistortion):
    def __init__(self, hue=0.2, saturation=1.5, exposure=1.5):
        self.hue = hue
        self.saturation = saturation
        self.exposure = exposure

    def random_distort(self, image, mode="bgr"):
        if mode == "bgr":
            flag_to_hsv = cv2.COLOR_BGR2HSV
            flag_from_hsv = cv2.COLOR_HSV2BGR
        elif mode == "rgb":
            flag_to_hsv = cv2.COLOR_RGB2HSV
            flag_from_hsv = cv2.COLOR_HSV2RGB
        else:
            raise ValueError("unrecognised color mode {}".format(mode))
        dhue = np.random.uniform(-self.hue, self.hue)
        dsat = self._rand_scale(self.saturation)
        dval = self._rand_scale(self.exposure)
        image_hsv = cv2.cvtColor(image, flag_to_hsv)
        image_hsv[:, :, 1] = cv2.multiply(image_hsv[:, :, 1], dsat)
        image_hsv[:, :, 2] = cv2.multiply(image_hsv[:, :, 2], dval)
        image_hsv = cv2.add(image_hsv, dhue)
        return cv2.cvtColor(image_hsv, flag_from_hsv)

    @staticmethod
    def _rand_scale(s):
        scale = np.random.uniform(1, s)
        if np.random.uniform(0, 1) < 0.5:
            return scale
        return 1 / scale


class GaussianBlur(IntensityDistortion):
    def __init__(self, prob=0.5, ksize=(5, 5)):
        self.prob = prob
        self.ksize = ksize

    def random_distort(self, image, mode="bgr"):
        if random.random() < self.prob:
            return cv2.GaussianBlur(image, self.ksize, 0)
        return image
