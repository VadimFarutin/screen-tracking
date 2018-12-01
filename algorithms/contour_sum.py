import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt

from util import get_screen_size, get_object_points, load_matrix,\
    load_positions, get_video_frame_size, format_params


class ContourSumTracker:
    EDGE_CONTROL_POINTS_NUMBER = 50
    EDGE_NUMBER = 4

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
            self.contour_gradient_sum_oriented, x0,
            (frame1_gradient_map, frame2_gradient_map,
             pos1_rotation, pos1_translation),
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

        success, frame = capture.read()
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gradient_map = cv2.Laplacian(current_gray_frame, cv2.CV_64F)

        for i in range(len(init_params) - 1):
            param1 = init_params[i]
            param2 = init_params[i + 1]

            success, frame = capture.read()
            next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            next_gradient_map = cv2.Laplacian(next_gray_frame, cv2.CV_64F)

            rmat1 = param1[0]
            rvec1, _ = cv2.Rodrigues(rmat1)
            tvec1 = param1[1]

            rmat2 = param2[0]
            rvec2, _ = cv2.Rodrigues(rmat2)
            tvec2 = param2[1]
            x = np.concatenate((rvec2, tvec2), axis=None)
            values = []

            for i in range(len(x)):
                x0 = np.copy(x)
                xi = steps * eps[i] + x0[i]
                f_values = []

                for j in range(len(xi)):
                    x0[i] = xi[j]
                    f_value = self.contour_gradient_sum_oriented(
                        x0,
                        current_gradient_map, next_gradient_map,
                        rvec1, tvec1)
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
            current_gradient_map = next_gradient_map

        capture.release()

    def contour_gradient_sum(self, x, image):
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

    def contour_gradient_sum_oriented(
            self, x, image1, image2, pos1_rotation, pos1_translation):
        rvec_res = np.array([[x[0]], [x[1]], [x[2]]])
        tvec_res = np.array([[x[3]], [x[4]], [x[5]]])

        image_points1, _ = cv2.projectPoints(
            np.array(self.control_points),
            pos1_rotation, pos1_translation, self.camera_mat, None)
        image_points1 = image_points1.reshape((len(self.control_points), 2))

        image_points2, _ = cv2.projectPoints(
            np.array(self.control_points),
            rvec_res, tvec_res, self.camera_mat, None)
        image_points2 = image_points2.reshape((len(self.control_points), 2))

        f_value = 0

        for i in range(ContourSumTracker.EDGE_NUMBER):
            gradient_sum1 = 0
            gradient_sum2 = 0

            for j in range(ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER):
                point_at_map1 = np.int32(image_points1[
                    i * ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER + j])  # + self.frame_size // 2
                if 0 <= point_at_map1[1] < self.frame_size[1] \
                        and 0 <= point_at_map1[0] < self.frame_size[0]:
                    gradient_sum1 += image1[point_at_map1[1],
                                            point_at_map1[0]]

                point_at_map2 = np.int32(image_points2[
                    i * ContourSumTracker.EDGE_CONTROL_POINTS_NUMBER + j])  # + self.frame_size // 2
                if 0 <= point_at_map2[1] < self.frame_size[1] \
                        and 0 <= point_at_map2[0] < self.frame_size[0]:
                    gradient_sum2 += image2[point_at_map2[1],
                                            point_at_map2[0]]

            if gradient_sum1 >= 0:
                f_value += gradient_sum2
            else:
                f_value += -gradient_sum2

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
