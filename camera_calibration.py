import sys
import numpy as np
import cv2

from util import get_screen_size, get_object_points, get_image_points,\
    center_points, get_video_frame_size


class CameraCalibrator:
    def __init__(self, test_path):
        self.test_path = test_path
        self.camera_matrix = None

    def calibrate_camera(self):
        width, height = get_screen_size(self.test_path + '/screen_parameters.csv')

        image_points_all = get_image_points(self.test_path + '/video.mp4')
        frame_size = get_video_frame_size(self.test_path + '/video.mp4')
        image_points_all = center_points(image_points_all, frame_size)

        object_points = get_object_points(width, height)
        object_points_all = [object_points for _ in image_points_all]

        flags = cv2.CALIB_ZERO_TANGENT_DIST \
            + cv2.CALIB_FIX_K1 \
            + cv2.CALIB_FIX_K2 \
            + cv2.CALIB_FIX_K3

        _, self.camera_matrix, _, _, _ = cv2.calibrateCamera(
            object_points_all, image_points_all, frame_size, None, None, flags=flags)

    def save_matrix(self):
        np.savetxt(self.test_path + '/camera_matrix.txt', self.camera_matrix)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Wrong number of arguments!')
        print('Expected format: python camera_calibration.py <test_path>')
        exit(1)

    calibrator = CameraCalibrator(args[1])
    calibrator.calibrate_camera()
    calibrator.save_matrix()
