import sys
import csv
import numpy as np
import cv2

from util import get_screen_size, get_object_points, get_image_points,\
    center_points, get_video_frame_size, load_matrix


class PositionsInitializer:
    def __init__(self, test_path):
        self.test_path = test_path
        self.extrinsic_params = None

    def get_extrinsic_params(self, object_points_all, image_points_all, camera_matrix):
        def solve_pnp(index, object_points, image_points):
            success, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, None)
            rmtx, _ = cv2.Rodrigues(rvec)
            params = np.concatenate((rmtx, tvec), axis=None)
            indexed_params = [index] + list(params.tolist())
            return indexed_params

        length = len(image_points_all)
        zipped_points = zip(range(length), object_points_all, image_points_all)

        extrinsic_params = [solve_pnp(i, object_points, image_points)
                            for i, object_points, image_points
                            in zipped_points]

        return extrinsic_params

    def save_positions(self):
        with open(self.test_path + '/positions.csv', 'w', newline='') as csv_file:
            positions_writer = csv.writer(csv_file, delimiter=',')
            for params in self.extrinsic_params:
                positions_writer.writerow(params)

    def init_positions(self):
        width, height = get_screen_size(self.test_path + '/screen_parameters.csv')

        image_points_all = get_image_points(self.test_path + '/video.mp4')
        frame_size = get_video_frame_size(self.test_path + '/video.mp4')
        image_points_all = center_points(image_points_all, frame_size)

        object_points = get_object_points(width, height)
        object_points_all = [object_points for _ in image_points_all]

        camera_matrix = load_matrix(self.test_path + '/camera_matrix.txt')

        self.extrinsic_params = self.get_extrinsic_params(
            object_points_all, image_points_all, camera_matrix)


if __name__ == '__main__':
    args = sys.argv
    if len(args) != 2:
        print('Wrong number of arguments!')
        print('Expected format: python init_positions.py <test_path>')
        exit(1)

    initializer = PositionsInitializer(args[1])
    initializer.init_positions()
    initializer.save_positions()
