import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from util import get_screen_size, get_object_points, load_matrix,\
    load_positions, get_video_frame_size, format_params, project_points_int,\
    is_point_in, rodrigues


class ContourSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 100
    EDGE_NUMBER = 4
    R_BOUNDS_EPS = 1e-1
    T_BOUNDS_EPS = 1

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
        frame1_gradient_map = cv2.Laplacian(frame1_grayscale_mat, cv2.CV_64F)
        frame2_gradient_map = cv2.Laplacian(frame2_grayscale_mat, cv2.CV_64F)

        pos1_rotation = rodrigues(pos1_rotation_mat)
        x0 = self.extrinsic_params_to_array(pos1_rotation, pos1_translation)
        step_eps = 1e-3
        bounds = ContourSumTracker.optimization_bounds(x0)
        gradient_sums1 = self.get_sides_gradient_sum(
            frame1_gradient_map, pos1_rotation, pos1_translation)

        # ans_vec = optimize.minimize(
        #     self.contour_gradient_sum, x0,
        #     frame2_gradient_map,
        #     bounds=bounds, options={'eps': step_eps})

        ans_vec = optimize.minimize(
            self.contour_gradient_sum_oriented, x0,
            (frame2_gradient_map, gradient_sums1),
            bounds=bounds, options={'eps': step_eps})

        pos2_rotation, pos2_translation = self.array_to_extrinsic_params(ans_vec.x)
        pos2_rotation_mat = rodrigues(pos2_rotation)
        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def optimization_bounds(x):
        bounds = ((x[0] - ContourSumTracker.R_BOUNDS_EPS, x[0] + ContourSumTracker.R_BOUNDS_EPS),
                  (x[1] - ContourSumTracker.R_BOUNDS_EPS, x[1] + ContourSumTracker.R_BOUNDS_EPS),
                  (x[2] - ContourSumTracker.R_BOUNDS_EPS, x[2] + ContourSumTracker.R_BOUNDS_EPS),
                  (x[3] - ContourSumTracker.T_BOUNDS_EPS, x[3] + ContourSumTracker.T_BOUNDS_EPS),
                  (x[4] - ContourSumTracker.T_BOUNDS_EPS, x[4] + ContourSumTracker.T_BOUNDS_EPS),
                  (x[5] - ContourSumTracker.T_BOUNDS_EPS, x[5] + ContourSumTracker.T_BOUNDS_EPS))
        return bounds

    def show_slices(self, video_path, init_params):
        capture = cv2.VideoCapture(video_path)
        half_number_of_steps = 100
        steps = np.arange(-half_number_of_steps, half_number_of_steps + 1, 1)
        eps = np.array([1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2])

        success, frame = capture.read()
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gradient_map = cv2.Laplacian(current_gray_frame, cv2.CV_64F)

        for k in range(len(init_params) - 1):
            success, frame = capture.read()
            next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_gradient_map = cv2.Laplacian(next_gray_frame, cv2.CV_64F)

            param1 = init_params[k]
            param2 = init_params[k + 1]

            rmat1 = param1[0]
            rvec1 = rodrigues(rmat1)
            tvec1 = param1[1]
            gradient_sums1 = self.get_sides_gradient_sum(
                current_gradient_map, rvec1, tvec1)

            rmat2 = param2[0]
            rvec2 = rodrigues(rmat2)
            tvec2 = param2[1]
            x_opt = self.extrinsic_params_to_array(rvec2, tvec2)
            values = []

            for i in range(len(x_opt)):
                x0 = np.copy(x_opt)
                xi = steps * eps[i] + x0[i]
                f_values = []

                for j in range(len(xi)):
                    x0[i] = xi[j]
                    # f_value = self.contour_gradient_sum(
                    #     x0,
                    #     next_gradient_map)
                    f_value = self.contour_gradient_sum_oriented(
                        x0, next_gradient_map, gradient_sums1)
                    f_values.append(-f_value)

                values.append((xi, f_values))

            x_prev = self.extrinsic_params_to_array(rvec1, tvec1)

            found_mat, found_tvec = self.track(
                current_gray_frame, next_gray_frame, rmat1, tvec1)
            found_rvec = rodrigues(found_mat)
            x_found = self.extrinsic_params_to_array(found_rvec, found_tvec)
            found_f_value = -self.contour_gradient_sum_oriented(
                x_found, next_gradient_map, gradient_sums1)

            f, axarr = plt.subplots(2, 3, figsize=(20, 10))

            for i in range(2):
                for j in range(3):
                    axarr[i][j].plot(
                        values[i * 3 + j][0], values[i * 3 + j][1],
                        color='blue')
                    axarr[i][j].plot(
                        [values[i * 3 + j][0][half_number_of_steps]],
                        [values[i * 3 + j][1][half_number_of_steps]],
                        'o', color='green')

                    bounds_eps = ContourSumTracker.R_BOUNDS_EPS \
                        if i == 0 \
                        else ContourSumTracker.T_BOUNDS_EPS

                    axarr[i][j].axvline(x=x_prev[i * 3 + j],
                                        color='blue', linestyle='dashed')
                    axarr[i][j].axvline(x=x_prev[i * 3 + j] - bounds_eps,
                                        color='blue', linestyle='dotted')
                    axarr[i][j].axvline(x=x_prev[i * 3 + j] + bounds_eps,
                                        color='blue', linestyle='dotted')

                    axarr[i][j].axvline(x=x_found[i * 3 + j],
                                        color='red', linestyle='dashed')
                    axarr[i][j].plot(
                        [x_found[i * 3 + j]], [found_f_value],
                        'o', color='red')

                    axarr[i][j].set_title('%s[%i]' %
                                          ('rvec' if i == 0 else 'tvec', j))

            plt.show()
            current_gray_frame = next_gray_frame
            current_gradient_map = next_gradient_map

        capture.release()

    def get_sides_gradient_sum(self, image, rvec, tvec):
        image_points = project_points_int(
            self.control_points, rvec, tvec, self.camera_mat)

        gradient_sums = []

        for i in range(ContourSumTracker.EDGE_NUMBER):
            gradient_sum = 0

            for j in range(ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER):
                point = image_points[
                    i * ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER + j]  # + self.frame_size // 2
                if is_point_in(point, self.frame_size):
                    gradient_sum += image[point[1], point[0]]

            gradient_sums.append(gradient_sum)

        return gradient_sums

    def contour_gradient_sum(self, x, image):
        rvec, tvec = self.array_to_extrinsic_params(x)
        gradient_sums = self.get_sides_gradient_sum(image, rvec, tvec)
        f_value = 0

        for gradient_sum in gradient_sums:
            f_value += abs(gradient_sum)

        return -f_value

    def contour_gradient_sum_oriented(self, x, image2, gradient_sums1):
        rvec2, tvec2 = self.array_to_extrinsic_params(x)
        gradient_sums2 = self.get_sides_gradient_sum(image2, rvec2, tvec2)

        f_value = 0

        for gradient_sum1, gradient_sum2 in zip(gradient_sums1, gradient_sums2):
            # if gradient_sum1 >= 0:
            #     f_value += gradient_sum2
            # else:
            #     f_value += -gradient_sum2
            # f_value += 1 / max(abs(gradient_sum1 - gradient_sum2), 1)
            f_value += -abs(gradient_sum1 - gradient_sum2)

        return -f_value

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

    def extrinsic_params_to_array(self, rvec, tvec):
        return np.concatenate((rvec, tvec), axis=None)

    def array_to_extrinsic_params(self, x):
        rvec = np.array([[x[0]], [x[1]], [x[2]]])
        tvec = np.array([[x[3]], [x[4]], [x[5]]])
        return rvec, tvec


if __name__ == '__main__':
    # test_path = '../tests/home_monitor'
    test_path = '../tests/living_room_on'

    camera_matrix = load_matrix(test_path + '/camera_matrix.txt')
    width, height = get_screen_size(test_path + '/screen_parameters.csv')
    object_points = get_object_points(width, height)
    loaded_params = load_positions(test_path + '/positions.csv')
    extrinsic_params = format_params(loaded_params)
    extrinsic_params = extrinsic_params[0:3]
    frame_size = get_video_frame_size(test_path + '/video.mp4')

    tracker = ContourSumTracker(camera_matrix, object_points, frame_size)
    tracker.show_slices(test_path + '/video.mp4', extrinsic_params)
