import cv2
from geometric_distortion import RandomTranslation, RandomScalingAndRotation, RandomShearing
from color_distortion import ColorDistorion, GaussianBlur
from augmentation import SequentialTransform


def draw_points(img, points, enclose=True):
    for p1, p2 in zip(points[:-1], points[1:]):
        p1 = int(p1[0]), int(p1[1])
        p2 = int(p2[0]), int(p2[1])
        cv2.line(img, p1, p2, color=(255, 255, 0), thickness=2)
    if enclose:
        p1, p2 = points[0], points[-1]
        p1 = int(p1[0]), int(p1[1])
        p2 = int(p2[0]), int(p2[1])
        cv2.line(img, p1, p2, color=(255, 255, 0), thickness=2)

    return img


if __name__ == '__main__':
    image = cv2.imread("frame560.jpg")
    h, w = image.shape[:2]
    image = cv2.resize(image, (w//3, h//3), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    points = [[471, 211], [505, 211], [505, 262], [472, 262]]
    clone = draw_points(image.copy(), points)

    translation = RandomTranslation((-0.1, 0.1), (-0.1, 0.1))
    rotation_and_scaling = RandomScalingAndRotation((-10, 10), (0.8, 1.2))
    shearing = RandomShearing((-0.1, 0.1), (-0.1, 0.1))
    color_distortion = ColorDistorion()
    blurring = GaussianBlur(0.5)
    sequential = SequentialTransform([translation, rotation_and_scaling, shearing],
                                     [color_distortion, blurring],
                                     (w, h))

    for i in range(10):
        img_aug, points_aug = sequential.apply_transform(image, points, border_mode=cv2.BORDER_CONSTANT)
        draw_points(img_aug, points_aug)
        # print(points_aug)
        cv2.imshow("original", clone)
        cv2.imshow("augimg", img_aug)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
