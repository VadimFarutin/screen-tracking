import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from util import get_screen_size, get_object_points, load_matrix,\
    load_positions, get_video_frame_size, format_params


class ContourSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 100

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))
        self.control_points, _ = ContourSumTracker.control_points(
            self.object_points, ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER)

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
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
            self.func, x0, frame2_grayscale_mat,
            bounds=bounds, options={'eps': step_eps})

        rvec = np.array([[ans_vec.x[0]], [ans_vec.x[1]], [ans_vec.x[2]]])
        tvec = np.array([[ans_vec.x[3]], [ans_vec.x[4]], [ans_vec.x[5]]])
        rmat, _ = cv2.Rodrigues(rvec)
        return rmat, tvec

    def show_slices(self, video_path, init_params):
        capture = cv2.VideoCapture(video_path)
        half_number_of_steps = 100
        steps = np.arange(-half_number_of_steps, half_number_of_steps + 1, 1)
        eps = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])

        for param in init_params:
            success, frame = capture.read()
            next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            rmat = param[0]
            rvec, _ = cv2.Rodrigues(rmat)
            tvec = param[1]
            x = np.concatenate((rvec, tvec), axis=None)
            values = []

            for i in range(len(x)):
                x0 = np.copy(x)
                xi = steps * eps[i] + x0[i]
                f_values = []

                for j in range(len(xi)):
                    x0[i] = xi[j]
                    f_value = self.func(x0, next_gray_frame)
                    f_values.append(-f_value)

                values.append((xi, f_values))

            f, axarr = plt.subplots(2, 3, figsize=(20, 10))

            for i in range(2):
                for j in range(3):
                    axarr[i][j].plot(values[i * 3 + j][0], values[i * 3 + j][1], color='blue')
                    # axarr[i][j].plot(values[i * 3 + j][0], values[i * 3 + j][1], 'o', color='blue')
                    axarr[i][j].plot([values[i * 3 + j][0][half_number_of_steps]],
                                     [values[i * 3 + j][1][half_number_of_steps]], 'o', color='red')
                    axarr[i][j].set_title('%s[%i]' % ('rvec' if i == 0 else 'tvec', j))

            plt.show()

        capture.release()

    def func(self, x, image):
        rvec_res = np.array([[x[0]], [x[1]], [x[2]]])
        tvec_res = np.array([[x[3]], [x[4]], [x[5]]])
        image_points, _ = cv2.projectPoints(
            np.array(self.control_points),
            rvec_res, tvec_res, self.camera_mat, None)
        image_points = image_points.reshape((len(self.control_points), 2))

        gradient_sum = 0

        for point in image_points:
            point_at_map = np.int32(point)  # + self.frame_size // 2
            if 0 <= point_at_map[1] < self.frame_size[1] \
                    and 0 <= point_at_map[0] < self.frame_size[0]:
                gradient_sum += abs(image[point_at_map[1],
                                          point_at_map[0]])

        return -gradient_sum

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


if __name__ == '__main__':
    # test_path = '../tests/home_monitor'
    test_path = '../tests/living_room'

    camera_matrix = load_matrix(test_path + '/camera_matrix.txt')
    width, height = get_screen_size(test_path + '/screen_parameters.csv')
    object_points = get_object_points(width, height)
    loaded_params = load_positions(test_path + '/positions.csv')
    extrinsic_params = format_params(loaded_params)
    extrinsic_params = extrinsic_params[0:10]
    frame_size = get_video_frame_size(test_path + '/video.mp4')

    tracker = ContourSumTracker(camera_matrix, object_points, frame_size)
    tracker.show_slices(test_path + '/video.mp4', extrinsic_params)
