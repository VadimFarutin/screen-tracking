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
    EDGE_CONTROL_POINTS_NUMBER = 20
    EDGE_NUMBER = 4
    ALPHA_BOUNDS_EPS = 1e-2
    T_BOUNDS_EPS = 10
    HALF_WINDOW_WIDTH = 20

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))

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
        found_corners = np.array([self.move_line(pair,
                                                 frame1_gradient_map,
                                                 frame2_gradient_map)
                                  for pair in corner_pairs])
        found_corners = np.append(found_corners, [found_corners[0]], axis=0)
        found_corner_pairs = np.array(list(zip(found_corners[:-1],
                                               found_corners[1:])))

        image_points = np.array([LineSumTracker.lines_intersection(pair[0], pair[1])
                                 for pair in found_corner_pairs])
        image_points = np.append([image_points[-1]], image_points, axis=0)
        image_points = image_points[:-1]

        _, pos2_rotation, pos2_translation = cv2.solvePnP(
            self.object_points, image_points, self.camera_mat, None)
        pos2_rotation_mat = rodrigues(pos2_rotation)
        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def optimization_bounds_t(x):
        bounds = [slice(x[0] - LineSumTracker.T_BOUNDS_EPS,
                        x[0] + LineSumTracker.T_BOUNDS_EPS,
                        2 * LineSumTracker.T_BOUNDS_EPS / 20),
                  slice(x[1] - LineSumTracker.T_BOUNDS_EPS,
                        x[1] + LineSumTracker.T_BOUNDS_EPS,
                        2 * LineSumTracker.T_BOUNDS_EPS / 20),
                  (x[2], x[2] + 1e-9, 1)]

        return bounds

    @staticmethod
    def optimization_bounds_alpha(x):
        bounds = [(x[0], x[0] + 1e-9, 1),
                  (x[1], x[1] + 1e-9, 1),
                  slice(x[2] - LineSumTracker.ALPHA_BOUNDS_EPS,
                        x[2] + LineSumTracker.ALPHA_BOUNDS_EPS,
                        2 * LineSumTracker.ALPHA_BOUNDS_EPS / 500)]

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
        corners = LineSumTracker.array_to_corners(x, length)
        gradient_sum2 = self.get_gradient_sum_for_side(corners, image2)
        signs = np.sign(gradient_sum1)
        # f_value = np.sum(signs * gradient_sum2)
        f_value = np.sum(-np.abs(gradient_sum1 - gradient_sum2))

        return -f_value

    @staticmethod
    def get_search_direction(tana):
        pi4 = math.pi / 4
        pi8 = pi4 / 2

        if math.fabs(tana) >= math.tan(pi4 + pi8):
            return np.array([1, 0])
        elif math.fabs(tana) <= math.tan(pi8):
            return np.array([0, 1])
        elif math.tan(pi8) < tana < math.tan(pi4 + pi8):
            return np.array([1, 1])
        elif -math.tan(pi8) > tana > -math.tan(pi4 + pi8):
            return np.array([1, -1])

    def get_window_gradients(self, image, middle_point, step):
        window_range = np.arange(-LineSumTracker.HALF_WINDOW_WIDTH,
                                 LineSumTracker.HALF_WINDOW_WIDTH + 1,
                                 1)
        window_range = window_range.reshape((len(window_range), 1))
        step = step.reshape((1, 2))
        steps = np.dot(window_range, step)
        image_points = steps + middle_point

        frame_size = self.frame_size
        binded_is_point_in = lambda point: is_point_in(point, frame_size)
        mask = np.apply_along_axis(binded_is_point_in, 1, image_points)
        image_points = image_points[mask]
        image_points_idx = image_points.T
        selected_gradients = image[image_points_idx[1], image_points_idx[0]]

        return selected_gradients

    def get_window_gradients_for_side(self, corners, step, image):
        image_points, _ = LineSumTracker.control_points(
            corners, LineSumTracker.EDGE_CONTROL_POINTS_NUMBER)
        image_points = np.int32(np.rint(image_points))
        binded_get_window_gradients = \
            lambda point: self.get_window_gradients(image, point, step)
        window_gradients = np.apply_along_axis(binded_get_window_gradients,
                                               1,
                                               image_points)

        return window_gradients

    def window_gradients_distance(
            self, x, image2, window_gradients1, step, length):
        corners = LineSumTracker.array_to_corners(x, length)
        window_gradients2 = self.get_window_gradients_for_side(
            corners, step, image2)
        distances = window_gradients1 - window_gradients2
        distances = np.apply_along_axis(np.abs, 1, distances)
        # def threshold(d):
        #     if d > 10:
        #         return 2
        #     if d > 5:
        #         return 1
        #     return 0
        # threshold_vect = np.vectorize(threshold)
        # distances = np.apply_along_axis(threshold_vect, 1, distances)
        distances = np.apply_along_axis(np.sum, 1, distances)
        f_value = -np.sum(distances)

        return -f_value

    def move_line(self, corners, frame1_gradient_map, frame2_gradient_map):
        gradient_sum1 = self.get_gradient_sum_for_side(
            corners, frame1_gradient_map)
        x0, length = LineSumTracker.corners_to_array(corners)
        tana = np.tan(x0[2])
        step = self.get_search_direction(tana)
        window_gradients1 = self.get_window_gradients_for_side(
            corners, step, frame1_gradient_map)
        bounds = LineSumTracker.optimization_bounds_t(x0)

        start = time.time()
        # ans_vec = optimize.brute(
        #     func=self.contour_gradient_sum_oriented,
        #     ranges=bounds,
        #     args=(frame2_gradient_map, gradient_sum1, length),
        #     finish=None,
        # )
        ans_vec = optimize.brute(
            func=self.window_gradients_distance,
            ranges=bounds,
            args=(frame2_gradient_map, window_gradients1, step, length),
            finish=None,
        )
        end = time.time()
        print(end - start)

        x0 = ans_vec
        bounds = LineSumTracker.optimization_bounds_alpha(x0)
        start = time.time()
        # ans_vec = optimize.brute(
        #     func=self.contour_gradient_sum_oriented,
        #     ranges=bounds,
        #     args=(frame2_gradient_map, gradient_sum1, length),
        #     finish=None,
        # )
        ans_vec = optimize.brute(
            func=self.window_gradients_distance,
            ranges=bounds,
            args=(frame2_gradient_map, window_gradients1, step, length),
            finish=None,
        )
        end = time.time()
        print(end - start)

        corners = LineSumTracker.array_to_corners(ans_vec, length)
        return corners

    @staticmethod
    def control_points(object_points, one_side_count):
        points = np.copy(object_points)

        control_points = [list(points[0] * (j + 1) / (one_side_count + 1)
                               + points[1] * (one_side_count - j) / (one_side_count + 1))
                          for j in range(one_side_count)]

        return control_points, None

    @staticmethod
    def corners_to_array(corners):
        diff = corners[1] - corners[0]
        center = corners[0] + diff / 2
        length = np.linalg.norm(diff)
        alpha = np.arcsin(diff[1] / length)
        if diff[0] <= 0:
            alpha = -alpha
        x = np.array([center[0], center[1], alpha])

        return x, length

    @staticmethod
    def array_to_corners(x, length):
        center = np.array([x[0], x[1]])
        alpha = x[2]
        sina = np.sin(alpha)
        cosa = np.cos(alpha)
        diff = np.array([cosa, sina]) * length
        corners = np.array([center - diff / 2, center + diff / 2])

        return corners

    @staticmethod
    def lines_intersection(first_line_points, second_line_points):
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
