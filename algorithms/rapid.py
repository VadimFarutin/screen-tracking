import cv2
import numpy as np
import math as math

from util import project_points_int, change_coordinate_system, rodrigues


class RapidScreenTracker:
    EDGE_CONTROL_POINTS_NUMBER = 100
    EDGE_NUMBER = 4
    EDGE_ERROR_THRESHOLD = 5
    EDGE_POINT_FILTER_THRESHOLD = 7
    NUMBER_OF_STEPS = 20
    # ALPHA = 1
    # BETA = 1e-1
    ALPHA = 1
    BETA = 0
    MIN_PNP_POINTS_ALLOWED = 3

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))
        self.vec_speed = np.array([[0], [0], [0], [0], [0], [0]],
                                  dtype=np.float32)

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        control_points, control_points_pair = self.control_points(
            self.object_points, RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER)
        control_points = [control_points[i * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER:
                                         (i + 1) * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                          for i in range(RapidScreenTracker.EDGE_NUMBER)]
        control_points_pair = [control_points_pair[i * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER:
                                                   (i + 1) * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                               for i in range(RapidScreenTracker.EDGE_NUMBER)]

        image_points = []
        image_points_idx = []
        edge_lines = []
        rejected_points = []

        pos1_rotation = rodrigues(pos1_rotation_mat)
        # shift = self.frame_size // 2
        frame2_gradient_map = cv2.Laplacian(frame2_grayscale_mat, cv2.CV_64F)

        for i in range(RapidScreenTracker.EDGE_NUMBER):
            # noinspection PyPep8Naming
            R = control_points[i]
            # noinspection PyPep8Naming
            S = control_points_pair[i]
            r = project_points_int(
                R, pos1_rotation, pos1_translation, self.camera_mat)
            s = project_points_int(
                S, pos1_rotation, pos1_translation, self.camera_mat)
            # r = change_coordinate_system(r, shift)
            # s = change_coordinate_system(s, shift)

            found_points, found_points_idx = self.search_edge(
                r, s, frame2_gradient_map, i)
            # found_points = change_coordinate_system(found_points, -shift)

            if len(found_points) == 0:
                continue

            corners = RapidScreenTracker.linear_regression(found_points)
            edge_lines.append(corners)
            error = RapidScreenTracker.find_edge_error(found_points, corners)

            if error > RapidScreenTracker.EDGE_ERROR_THRESHOLD:
                continue

            accepted, accepted_idx, rejected = RapidScreenTracker.filter_edge_points(
                found_points, found_points_idx, corners)
            image_points.extend(accepted)
            image_points_idx.extend(accepted_idx)
            rejected_points.extend(rejected)

        image_points = np.array(image_points, np.float32)
        all_control_points = np.array([control_points[i // RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                                                     [i % RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                                      for i in image_points_idx])

        last_r_vec = np.copy(pos1_rotation)
        last_t_vec = np.copy(pos1_translation)

        if len(image_points) < RapidScreenTracker.MIN_PNP_POINTS_ALLOWED:
            rvec = np.copy(pos1_rotation)
            tvec = np.copy(pos1_translation)
        else:
            _, rvec, tvec = cv2.solvePnP(all_control_points, image_points,
                                         self.camera_mat, None,
                                         pos1_rotation, pos1_translation,
                                         useExtrinsicGuess=True)

        # retval, rvec, tvec, _ = cv2.solvePnPRansac(
        #     all_control_points, image_points, cameraMatrix, None)

        diff_r_vec = rvec - last_r_vec
        diff_t_vec = tvec - last_t_vec

        rvec = last_r_vec + RapidScreenTracker.ALPHA * diff_r_vec \
            + self.vec_speed[0:3]
        tvec = last_t_vec + RapidScreenTracker.ALPHA * diff_t_vec \
            + self.vec_speed[3:6]
        self.vec_speed += RapidScreenTracker.BETA \
            * np.append(rvec - last_r_vec, tvec - last_t_vec, axis=0)

        rmat = rodrigues(rvec)
        return rmat, tvec

    @staticmethod
    def get_search_direction(tana):
        pi4 = math.pi / 4
        pi8 = pi4 / 2

        if math.fabs(tana) >= math.tan(pi4 + pi8):
            return 1, 0
        elif math.fabs(tana) <= math.tan(pi8):
            return 0, 1
        elif math.tan(pi8) < tana < math.tan(pi4 + pi8):
            return 1, 1
        elif -math.tan(pi8) > tana > -math.tan(pi4 + pi8):
            return 1, -1

    @staticmethod
    def get_distance(tana, sina, cosa, n):
        pi4 = math.pi / 4
        pi8 = pi4 / 2

        if math.fabs(tana) >= math.tan(pi4 + pi8):
            return -n * sina
        elif math.fabs(tana) <= math.tan(pi8):
            return n * cosa
        elif math.tan(pi8) < tana < math.tan(pi4 + pi8):
            return n * (cosa - sina)
        elif -math.tan(pi8) > tana > -math.tan(pi4 + pi8):
            return n * (cosa + sina)

    def search_edge(self, r, s, gradient_map, edge_index):
        cos_a = np.array([(si[0] - ri[0]) / np.linalg.norm(si - ri)
                          for si, ri in zip(s, r)])
        sin_a = np.array([(si[1] - ri[1]) / np.linalg.norm(si - ri)
                          for si, ri in zip(s, r)])
        tana = sin_a / cos_a

        found_points = []
        found_points_idx = []

        for j in range(RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER):
            step_x, step_y = RapidScreenTracker.get_search_direction(tana[j])

            if not 0 <= r[j][0] < self.frame_size[0] \
                    or not 0 <= r[j][1] < self.frame_size[1]:
                continue

            point = self.search_edge_from_point(
                gradient_map, r[j], (step_x, step_y))
            found_points.append(point)
            found_points_idx.append(
                edge_index * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER + j)

        return found_points, found_points_idx

    def search_edge_from_point(self, edges, start, step):
        max_gradient_point = np.copy(start)
        max_gradient = abs(edges[start[1], start[0]])

        max_gradient, max_gradient_point = \
            self.search_edge_from_point_to_one_side(
                edges, start, step, RapidScreenTracker.NUMBER_OF_STEPS,
                max_gradient, max_gradient_point)
        step = (-step[0], -step[1])
        max_gradient, max_gradient_point = \
            self.search_edge_from_point_to_one_side(
                edges, start, step, RapidScreenTracker.NUMBER_OF_STEPS,
                max_gradient, max_gradient_point)

        return max_gradient_point

    def search_edge_from_point_to_one_side(
            self, edges, start, step, count, max_gradient, max_gradient_point):
        current = np.copy(start)

        for i in range(count):
            current += step

            if not 0 <= current[0] < self.frame_size[1] \
                    or not 0 <= current[1] < self.frame_size[0]:
                continue

            gradient_value = abs(edges[current[1], current[0]])

            if max_gradient < gradient_value:
                max_gradient = gradient_value
                max_gradient_point = np.copy(current)

        return max_gradient, max_gradient_point

    @staticmethod
    def linear_regression(points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        if abs(x[0] - x[-1]) >= abs(y[0] - y[-1]):
            a = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(a, y, rcond=None)[0]
            corners = np.array([[x[0], k * x[0] + b],
                                [x[-1], k * x[-1] + b]])
        else:
            a = np.vstack([y, np.ones(len(y))]).T
            k, b = np.linalg.lstsq(a, x, rcond=None)[0]
            corners = np.array([[k * y[0] + b, y[0]],
                                [k * y[-1] + b, y[-1]]])

        return corners

    @staticmethod
    def find_edge_error(found_points, corners):
        error = 0

        for point in found_points:
            error += np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[0] - point)) \
                / np.linalg.norm(corners[1] - corners[0])
        error /= RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER

        return error

    @staticmethod
    def filter_edge_points(found_points, found_points_idx, corners):
        accepted = []
        accepted_idx = []
        rejected = []

        for point, j in zip(found_points, found_points_idx):
            d = np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[0] - point)) \
                / np.linalg.norm(corners[1] - corners[0])
            if d <= RapidScreenTracker.EDGE_POINT_FILTER_THRESHOLD:
                accepted.append(point)
                accepted_idx.append(j)
            else:
                rejected.append(point)

        return accepted, accepted_idx, rejected

    @staticmethod
    def control_points(object_points, one_side_count):
        points = np.copy(object_points)
        points = np.append(points, [object_points[0]], axis=0)

        control_points = [list(point * (j + 1) / (one_side_count + 1)
                          + next_point * (one_side_count - j) / (one_side_count + 1))
                          for (point, next_point) in zip(points[:-1], points[1:])
                          for j in range(one_side_count)]
        control_point_pairs = [point
                               for point in points[:-1]
                               for _ in range(one_side_count)]

        return control_points, control_point_pairs
