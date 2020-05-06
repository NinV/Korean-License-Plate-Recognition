import cv2
import numpy as np


class SequentialTransform:
    def __init__(self, geometric_transforms, color_distortions, out_size):
        self.geometric_transforms = geometric_transforms
        self.color_distortions = color_distortions
        self.out_size = out_size

    def _get_transformation_matrix(self, img_size):
        w, h = img_size
        T = np.identity(3)
        for transform in self.geometric_transforms:
            T = np.matmul(transform.get_transformation_matrix((w, h)), T)
        return T

    def apply_transform(self, image: np.ndarray, points=None, mode="bgr",
                        interpolation=cv2.INTER_AREA,
                        border_mode=cv2.BORDER_CONSTANT,
                        border_value=(127, 127, 127)):
        """
        :param image: numpy array
        :param points: list of 2D point.
        :return:
        """
        h, w = image.shape[:2]
        T = self._get_transformation_matrix(img_size=(w, h))
        out = cv2.warpPerspective(image.copy(), T, self.out_size, None, interpolation, border_mode, border_value)
        if points is not None:
            points = np.array(points, dtype=np.float)

            # convert to homogeneous coordinates
            if points.shape[1] == 2:
                nums = points.shape[0]
                points = np.hstack((points, np.ones((nums, 1), dtype=np.float)))
            points = np.matmul(T, points.T).T
            points = points[:, :2].tolist()

        for color_distortion in self.color_distortions:
            out = color_distortion.random_distort(out, mode)

        return out, points


