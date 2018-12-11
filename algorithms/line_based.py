import cv2
import numpy as np
import time
import math
from scipy import optimize
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from util import get_screen_size, get_object_points, load_matrix, \
    load_positions, get_video_frame_size, format_params, project_points_int, \
    is_point_in, rodrigues


class LineSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 10
    EDGE_NUMBER = 4
    ALPHA_BOUNDS_EPS = math.pi / 2
    T_BOUNDS_EPS = 5

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        frame1_gradient_map = cv2.Laplacian(frame1_grayscale_mat, cv2.CV_64F)
        frame2_gradient_map = cv2.Laplacian(frame2_grayscale_mat, cv2.CV_64F)
        pos1_rotation = rodrigues(pos1_rotation_mat)

        corners = project_points_int(self.object_points,
                                     pos1_rotation, pos1_translation,
                                     self.camera_mat)
        corners = np.append(corners, [corners[0]], axis=0)
        corner_pairs = np.array(list(zip(corners[:-1], corners[1:])))
        found_corners = np.array([self.move_line(pair, frame1_gradient_map, frame2_gradient_map)
                                  for pair in corner_pairs])
        found_corners = np.append(found_corners, [found_corners[0]], axis=0)
        found_corner_pairs = np.array(list(zip(found_corners[:-1], found_corners[1:])))

        image_points = np.array([self.lines_intersection(pair[0], pair[1])
                                 for pair in found_corner_pairs])
        image_points = np.append([image_points[-1]], image_points, axis=0)
        image_points = image_points[:-1]

        _, pos2_rotation, pos2_translation = cv2.solvePnP(
            self.object_points, image_points, self.camera_mat, None)
        pos2_rotation_mat = rodrigues(pos2_rotation)
        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def optimization_bounds1(x):
        bounds = [slice(x[0] - LineSumTracker.T_BOUNDS_EPS, x[0] + LineSumTracker.T_BOUNDS_EPS,
                        2 * LineSumTracker.T_BOUNDS_EPS / 100),
                  slice(x[1] - LineSumTracker.T_BOUNDS_EPS, x[1] + LineSumTracker.T_BOUNDS_EPS,
                        2 * LineSumTracker.T_BOUNDS_EPS / 100),
                  (x[2], x[2] + 1e-9, 1)]

        return bounds

    @staticmethod
    def optimization_bounds2(x):
        bounds = [(x[0], x[0] + 1e-9, 1),
                  (x[1], x[1] + 1e-9, 1),
                  slice(x[2] - LineSumTracker.ALPHA_BOUNDS_EPS,
                        x[2] + LineSumTracker.ALPHA_BOUNDS_EPS,
                        2 * LineSumTracker.ALPHA_BOUNDS_EPS / 50)]

        return bounds

    def get_gradient_sum(self, image, image_points):
        frame_size = self.frame_size
        binded_is_point_in = lambda point: is_point_in(point, frame_size)
        mask = np.apply_along_axis(binded_is_point_in, 1, image_points)
        image_points = image_points[mask]
        image_points_idx = image_points.T
        selected_gradients = image[image_points_idx[1], image_points_idx[0]]
        gradient_sum = sum(selected_gradients)

        return gradient_sum

    def get_gradient_sum_for_side(self, corners, image):
        image_points, _ = LineSumTracker.control_points(
            corners, LineSumTracker.EDGE_CONTROL_POINTS_NUMBER)
        image_points = np.int32(np.rint(image_points))
        gradient_sum = self.get_gradient_sum(image, image_points)

        return gradient_sum

    def contour_gradient_sum_oriented(self, x, image2, gradient_sum1, length):
        corners = self.array_to_corners(x, length)
        gradient_sum2 = self.get_gradient_sum_for_side(corners, image2)
        signs = np.sign(gradient_sum1)
        f_value = np.sum(signs * gradient_sum2)
        # f_value = np.sum(-np.abs(gradient_sums1 - gradient_sums2))

        return -f_value

    def move_line(self, corners, frame1_gradient_map, frame2_gradient_map):
        gradient_sum1 = self.get_gradient_sum_for_side(corners, frame1_gradient_map)
        x0, length = self.corners_to_array(corners)
        bounds = LineSumTracker.optimization_bounds1(x0)

        start = time.time()
        ans_vec = optimize.brute(
            func=self.contour_gradient_sum_oriented,
            ranges=bounds,
            args=(frame2_gradient_map, gradient_sum1, length),
        )
        end = time.time()
        print(end - start)

        x0 = ans_vec
        bounds = LineSumTracker.optimization_bounds2(x0)
        start = time.time()
        ans_vec = optimize.brute(
            func=self.contour_gradient_sum_oriented,
            ranges=bounds,
            args=(frame2_gradient_map, gradient_sum1, length),
        )
        end = time.time()
        print(end - start)

        corners = self.array_to_corners(ans_vec, length)
        return corners

    @staticmethod
    def control_points(object_points, one_side_count):
        points = np.copy(object_points)
        points = np.append(points, [object_points[0]], axis=0)
        point_pairs = np.array(list(zip(points[:-1], points[1:])))

        control_points = [list(pair[0] * (j + 1) / (one_side_count + 1)
                               + pair[1]
                                       * (one_side_count - j) / (one_side_count + 1))
                          for pair in point_pairs
                          for j in range(one_side_count)]
        # control_point_pairs = [point
        #                        for point in points[:-1]
        #                        for j in range(one_side_count)]

        return control_points, None

    def corners_to_array(self, corners):
        diff = corners[1] - corners[0]
        center = corners[0] + diff / 2
        length = np.linalg.norm(diff)
        alpha = np.arcsin(diff[1] / length)
        if diff[0] <= 0:
            alpha = -alpha
        x = np.array([center[0], center[1], alpha])

        return x, length

    def array_to_corners(self, x, length):
        center = np.array([x[0], x[1]])
        alpha = x[2]
        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        diff = np.array([cosa, sina]) * length
        corners = np.array([center - diff / 2, center + diff / 2])

        return corners

    def lines_intersection(self, first_line_points, second_line_points):
        A = np.array([[first_line_points[0][1] - first_line_points[1][1],
                       first_line_points[1][0] - first_line_points[0][0]],
                      [second_line_points[0][1] - second_line_points[1][1],
                       second_line_points[1][0] - second_line_points[0][0]]])
        B = np.array([first_line_points[1][0] * first_line_points[0][1]
                      - first_line_points[0][0] * first_line_points[1][1],
                      second_line_points[1][0] * second_line_points[0][1]
                      - second_line_points[0][0] * second_line_points[1][1]])
        point = np.linalg.solve(A, B)

        return point


if __name__ == '__main__':
    # test_path = '../tests/home_monitor'
    test_path = '../tests/living_room_on'

    camera_matrix = load_matrix(test_path + '/camera_matrix.txt')
    width, height = get_screen_size(test_path + '/screen_parameters.csv')
    object_points = get_object_points(width, height)
    loaded_params = load_positions(test_path + '/positions.csv')
    extrinsic_params = format_params(loaded_params)
    extrinsic_params = extrinsic_params[0:2]
    frame_size = get_video_frame_size(test_path + '/video.mp4')

    tracker = LineSumTracker(camera_matrix, object_points, frame_size)
    # tracker.show_slices(test_path + '/video.mp4', extrinsic_params)
