import cv2
import numpy as np
from scipy import optimize


class ContourSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 100

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        control_points, _ = ContourSumTracker.control_points(
            self.object_points, ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER)

        def func(x):
            rvec_res = np.array([[x[0]], [x[1]], [x[2]]])
            tvec_res = np.array([[x[3]], [x[4]], [x[5]]])
            image_points, _ = cv2.projectPoints(
                np.array(control_points),
                rvec_res, tvec_res, self.camera_mat, None)
            image_points = image_points.reshape((len(control_points), 2))

            gradient_sum = 0

            for point in image_points:
                point_at_map = np.int32(point) + self.frame_size // 2
                if 0 <= point_at_map[1] < self.frame_size[1] \
                        and 0 <= point_at_map[0] < self.frame_size[0]:
                    gradient_sum += abs(frame2_grayscale_mat[point_at_map[1],
                                                             point_at_map[0]])

            return -gradient_sum

        pos1_rotation, _ = cv2.Rodrigues(pos1_rotation_mat)
        x0 = np.concatenate((pos1_rotation, pos1_translation), axis=None)
        step_eps = 1e-3
        r_bounds_eps = 1
        t_bounds_eps = 5
        bounds = ((x0[0] - r_bounds_eps, x0[0] + r_bounds_eps),
                  (x0[1] - r_bounds_eps, x0[1] + r_bounds_eps),
                  (x0[2] - r_bounds_eps, x0[2] + r_bounds_eps),
                  (x0[3] - t_bounds_eps, x0[3] + t_bounds_eps),
                  (x0[4] - t_bounds_eps, x0[4] + t_bounds_eps),
                  (x0[5] - t_bounds_eps, x0[5] + t_bounds_eps))
        ans_vec = optimize.minimize(
            func, x0, bounds=bounds, options={'eps': step_eps})

        rvec = np.array([[ans_vec.x[0]], [ans_vec.x[1]], [ans_vec.x[2]]])
        tvec = np.array([[ans_vec.x[3]], [ans_vec.x[4]], [ans_vec.x[5]]])
        rmat, _ = cv2.Rodrigues(rvec)
        return rmat, tvec

    @staticmethod
    def control_points(object_points, one_side_count):
        points = np.copy(object_points)
        points = np.append(points, [object_points[0]], axis=0)

        control_points = [list(point * (j + 1) / (one_side_count + 1)
                               + nextPoint
                                     * (one_side_count - j) / (one_side_count + 1))
                          for (point, nextPoint) in zip(points[:-1], points[1:])
                          for j in range(one_side_count)]
        control_point_pairs = [point
                               for point in points[:-1]
                               for j in range(one_side_count)]

        return control_points, control_point_pairs
