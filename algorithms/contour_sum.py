import cv2
import numpy as np
import time
from scipy import optimize
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt

from util import get_screen_size, get_object_points, load_matrix, \
    load_positions, get_video_frame_size, format_params, project_points_int, \
    is_point_in, rodrigues, project_points


class ContourSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 10
    EDGE_NUMBER = 4
    R_BOUNDS_EPS = 3e-2
    T_BOUNDS_EPS = 1.7e0
    IMAGE_SIGMA = 0
    GRADIENT_SIGMA = 0
    SLICES = np.arange(1, EDGE_NUMBER) * EDGE_CONTROL_POINTS_NUMBER

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))
        self.control_points, _ = ContourSumTracker.control_points(
            self.object_points, ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER)

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        # frame1_grayscale_mat = gaussian_filter(frame1_grayscale_mat, ContourSumTracker.IMAGE_SIGMA)
        # frame2_grayscale_mat = gaussian_filter(frame2_grayscale_mat, ContourSumTracker.IMAGE_SIGMA)

        frame1_gradient_map = cv2.Laplacian(frame1_grayscale_mat, cv2.CV_64F)
        frame2_gradient_map = cv2.Laplacian(frame2_grayscale_mat, cv2.CV_64F)

        # frame1_gradient_map = gaussian_filter(frame1_gradient_map, ContourSumTracker.GRADIENT_SIGMA)
        # frame2_gradient_map = gaussian_filter(frame2_gradient_map, ContourSumTracker.GRADIENT_SIGMA)

        pos1_rotation = rodrigues(pos1_rotation_mat)
        x0 = self.extrinsic_params_to_array(pos1_rotation, pos1_translation)
        # step_eps = 1e-3
        bounds = ContourSumTracker.optimization_bounds_t(x0)
        gradient_sums1 = self.get_gradient_sum_for_sides(
            frame1_gradient_map, pos1_rotation, pos1_translation)

        # def take_step(x):
        #     # x[0:3] += np.random.uniform(-step_eps, step_eps, 3)
        #     # x[3:6] += np.random.uniform(-10 * step_eps, 10 * step_eps, 3)
        #     x[0:3] += [step_eps * random.randint(-1, 1) for i in range(3)]
        #     x[3:6] += [10 * step_eps * random.randint(-1, 1) for i in range(3)]
        #     return x
        #
        # def check_bounds(f_new, x_new, f_old, x_old):
        #     for xi, bound in zip(x_new, bounds):
        #         if not bound[0] <= xi <= bound[1]:
        #             return False
        #     return True
        #
        # ans_vec = optimize.minimize(
        #     self.contour_gradient_sum, x0,
        #     frame2_gradient_map,
        #     bounds=bounds, options={'eps': step_eps})
        #
        # ans_vec = optimize.minimize(
        #     fun=self.contour_gradient_sum_oriented,
        #     x0=x0,
        #     args=(frame2_gradient_map, gradient_sums1),
        #     method='TNC',
        #     bounds=bounds,
        #     options={
        #         'eps': step_eps,
        #         # 'disp': True,
        #         # 'maxcor': 10,
        #         # 'maxls': 10
        #         # 'maxiter': 5000,
        #         # 'maxfev': 5000,
        #         # 'adaptive': True,
        #     }
        # )
        start = time.time()
        ans_vec = optimize.brute(
            func=self.contour_gradient_sum_oriented,
            ranges=bounds,
            args=(frame2_gradient_map, gradient_sums1),
            # Ns=2,
            # full_output=True,
            # disp=True
        )
        end = time.time()
        print(end - start)

        x0 = ans_vec
        bounds = ContourSumTracker.optimization_bounds_r(x0)
        start = time.time()
        ans_vec = optimize.brute(
            func=self.contour_gradient_sum_oriented,
            ranges=bounds,
            args=(frame2_gradient_map, gradient_sums1),
            # Ns=2,
            # full_output=True,
            # disp=True
        )
        end = time.time()
        print(end - start)

        # ans_vec = optimize.basinhopping(
        #     func=self.contour_gradient_sum_oriented,
        #     x0=x0,
        #     niter=500,
        #     stepsize=10 * step_eps,
        #     minimizer_kwargs={'args': (frame2_gradient_map, gradient_sums1)},
        #     accept_test=check_bounds,
        #     # take_step=take_step
        # )

        pos2_rotation, pos2_translation = self.array_to_extrinsic_params(ans_vec)
        pos2_rotation_mat = rodrigues(pos2_rotation)
        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def optimization_bounds_t(x):
        bounds = [(x[0], x[0] + 1e-9, 1),
                  (x[1], x[1] + 1e-9, 1),
                  (x[2], x[2] + 1e-9, 1),
                  slice(x[3] - ContourSumTracker.T_BOUNDS_EPS,
                        x[3] + ContourSumTracker.T_BOUNDS_EPS,
                        2 * ContourSumTracker.T_BOUNDS_EPS / 45),
                  slice(x[4] - ContourSumTracker.T_BOUNDS_EPS,
                        x[4] + ContourSumTracker.T_BOUNDS_EPS,
                        2 * ContourSumTracker.T_BOUNDS_EPS / 45),
                  slice(x[5] - ContourSumTracker.T_BOUNDS_EPS,
                        x[5] + ContourSumTracker.T_BOUNDS_EPS,
                        2 * ContourSumTracker.T_BOUNDS_EPS / 45)]

        return bounds

    @staticmethod
    def optimization_bounds_r(x):
        bounds = [slice(x[0] - ContourSumTracker.R_BOUNDS_EPS,
                        x[0] + ContourSumTracker.R_BOUNDS_EPS,
                        2 * ContourSumTracker.R_BOUNDS_EPS / 20),
                  slice(x[1] - ContourSumTracker.R_BOUNDS_EPS,
                        x[1] + ContourSumTracker.R_BOUNDS_EPS,
                        2 * ContourSumTracker.R_BOUNDS_EPS / 20),
                  slice(x[2] - ContourSumTracker.R_BOUNDS_EPS,
                        x[2] + ContourSumTracker.R_BOUNDS_EPS,
                        2 * ContourSumTracker.R_BOUNDS_EPS / 20),
                  (x[3], x[3] + 1e-9, 1),
                  (x[4], x[4] + 1e-9, 1),
                  (x[5], x[5] + 1e-9, 1)]

        return bounds

    def show_slices(self, video_path, init_params):
        capture = cv2.VideoCapture(video_path)
        half_number_of_steps = 100
        steps = np.arange(-half_number_of_steps, half_number_of_steps + 1, 1)
        step_size = np.repeat([1e-3, 1e-2], [3, 3])

        success, frame = capture.read()
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # current_gray_frame = gaussian_filter(current_gray_frame, ContourSumTracker.IMAGE_SIGMA)
        current_gradient_map = cv2.Laplacian(current_gray_frame, cv2.CV_64F)
        # current_gradient_map = gaussian_filter(current_gradient_map, ContourSumTracker.GRADIENT_SIGMA)

        for k in range(len(init_params) - 1):
            success, frame = capture.read()
            next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # next_gray_frame = gaussian_filter(next_gray_frame, ContourSumTracker.IMAGE_SIGMA)
            next_gradient_map = cv2.Laplacian(next_gray_frame, cv2.CV_64F)
            # next_gradient_map = gaussian_filter(next_gradient_map, ContourSumTracker.GRADIENT_SIGMA)

            param1 = init_params[k]
            param2 = init_params[k + 1]

            rmat1 = param1[0]
            rvec1 = rodrigues(rmat1)
            tvec1 = param1[1]
            gradient_sums1 = self.get_gradient_sum_for_sides(
                current_gradient_map, rvec1, tvec1)

            rmat2 = param2[0]
            rvec2 = rodrigues(rmat2)
            tvec2 = param2[1]
            x_opt = self.extrinsic_params_to_array(rvec2, tvec2)
            values = []

            for i in range(len(x_opt)):
                x0 = np.copy(x_opt)
                xi = steps * step_size[i] + x0[i]
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
            prev_f_value = -self.contour_gradient_sum_oriented(
                x_prev, next_gradient_map, gradient_sums1)

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
                    axarr[i][j].axvline(x=values[i * 3 + j][0][half_number_of_steps],
                                        color='green', linestyle='dashed')

                    bounds_eps = ContourSumTracker.R_BOUNDS_EPS \
                        if i == 0 \
                        else ContourSumTracker.T_BOUNDS_EPS

                    axarr[i][j].axvline(x=x_prev[i * 3 + j],
                                        color='blue', linestyle='dashed')
                    axarr[i][j].plot(
                        [x_prev[i * 3 + j]], [prev_f_value],
                        'o', color='blue')
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

    def get_gradient_sum(self, image, image_points):
        frame_size = self.frame_size
        binded_is_point_in = lambda point: is_point_in(point, frame_size)
        mask = np.apply_along_axis(binded_is_point_in, 1, image_points)
        image_points = image_points[mask]
        image_points_idx = image_points.T
        selected_gradients = image[image_points_idx[1], image_points_idx[0]]
        gradient_sum = sum(selected_gradients)

        return gradient_sum

    def get_gradient_sum_for_sides(self, image, rvec, tvec):
        image_points = project_points_int(
            self.control_points, rvec, tvec, self.camera_mat)

        image_points_by_side = np.split(image_points, ContourSumTracker.SLICES)
        get_gradient_sum = self.get_gradient_sum
        gradient_sums = np.array([get_gradient_sum(image, points)
                                  for points in image_points_by_side])

        return gradient_sums

    def contour_gradient_sum(self, x, image):
        rvec, tvec = self.array_to_extrinsic_params(x)
        gradient_sums = self.get_gradient_sum_for_sides(image, rvec, tvec)
        f_value = np.sum(np.abs(gradient_sums))

        return -f_value

    def contour_gradient_sum_oriented(self, x, image2, gradient_sums1):
        rvec2, tvec2 = self.array_to_extrinsic_params(x)
        gradient_sums2 = self.get_gradient_sum_for_sides(image2, rvec2, tvec2)
        signs = np.sign(gradient_sums1)
        f_value = np.sum(signs * gradient_sums2)
        # f_value = np.sum(-np.abs(gradient_sums1 - gradient_sums2))

        return -f_value

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

    @staticmethod
    def extrinsic_params_to_array(rvec, tvec):
        return np.concatenate((rvec, tvec), axis=None)

    @staticmethod
    def array_to_extrinsic_params(x):
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
    extrinsic_params = extrinsic_params[0:5]
    frame_size = get_video_frame_size(test_path + '/video.mp4')

    tracker = ContourSumTracker(camera_matrix, object_points, frame_size)
    tracker.show_slices(test_path + '/video.mp4', extrinsic_params)
