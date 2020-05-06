import random
import numpy as np
import cv2


class GeometricDistortion:
    """
    Base class for all geometric distortion method
    """
    def _validate_input(self, *args):
        raise NotImplemented

    def get_transformation_matrix(self, img_size):
        raise NotImplemented


class RandomTranslation(GeometricDistortion):
    def __init__(self, tx_range, ty_range):
        self._validate_input(tx_range, ty_range)
        self.tx_range = tx_range
        self.ty_range = ty_range

    def _validate_input(self, *args):
        for arg in args:
            if len(arg) != 2:
                raise ValueError("Both tx_range and ty_range must have length of 2")
            min_value = min(arg)
            max_value = max(arg)
            if min_value < -1.:
                raise ValueError("translation range must not < -1")

            if max_value > 1.:
                raise ValueError("translation range must not > 1")

    def get_transformation_matrix(self, img_size):
        iw, ih = img_size
        tx = random.uniform(*self.tx_range) * iw
        ty = random.uniform(*self.tx_range) * ih

        T = np.array([[1, 0, tx],
                      [0, 1, ty],
                      [0, 0, 1]])
        return T


class RandomScalingAndRotation(GeometricDistortion):
    def __init__(self, angle_range, scale_range, center=(0, 0)):
        """
        :param angle_range: angle range in degree
        :param scale_range: scale range
        :param center: center point. Default: (0, 0)
        """
        self._validate_input(angle_range, scale_range)
        self.angle_range = angle_range
        self.scale_range = scale_range
        self.center = center

    def _validate_input(self, *args):
        angle_range, scale_range = args
        if len(angle_range) != 2:
            raise ValueError("angle_range must have length of 2")

        if len(scale_range) != 2:
            raise ValueError("scale_range must have length of 2")

        for value in scale_range:
            if value < 0:
                raise ValueError("scale_range must not < 0")

    def get_transformation_matrix(self, img_size):
        iw, ih = img_size
        angle = random.uniform(*self.angle_range)
        scale = random.uniform(*self.scale_range)

        T = cv2.getRotationMatrix2D(self.center, angle, scale)
        T = np.vstack((T, np.array([[0, 0, 1]])))
        return T


class RandomShearing(GeometricDistortion):
    def __init__(self, vertical_shearing_range, horizontal_shearing_range):
        self._validate_input(vertical_shearing_range, horizontal_shearing_range)
        self.vertical_shearing_range = vertical_shearing_range
        self.horizontal_shearing_range = horizontal_shearing_range

    def _validate_input(self, *args):
        v_range, h_range = args
        if len(v_range) != 2:
            raise ValueError("angle_range must have length of 2")

        if len(h_range) != 2:
            raise ValueError("scale_range must have length of 2")

    def get_transformation_matrix(self, img_size):
        iw, ih = img_size
        v_shearing = random.uniform(*self.vertical_shearing_range)
        h_shearing = random.uniform(*self.horizontal_shearing_range)

        # using this T matrix instead of vanilla one the ensure the mean of transformation is at image's center
        T = np.array([[1, h_shearing / 2, h_shearing * ih / 2],
                      [v_shearing / 2, 1, v_shearing * iw / 2],
                      [0, 0, 1]])
        return T
