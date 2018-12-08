import cv2
import numpy as np
from matplotlib import pyplot as plt

from algorithms.fake import FakeTracker
from algorithms.rapid import RapidScreenTracker
from algorithms.contour_sum import ContourSumTracker
from algorithms.sift import SiftTracker
from algorithms.line_based import LineSumTracker

from util import get_screen_size, get_object_points, load_matrix,\
    load_positions, get_video_frame_size, format_params


class Testing:
    def __init__(self):
        pass

    def run_algorithm(self, algorithm_impl, video_path, init_params):
        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        current_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        estimated_extrinsic_params = [init_params[0]]

        for i in range(len(init_params) - 1):
            success, frame = capture.read()
            next_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if not success:
                break

            rmat, tvec = algorithm_impl.track(
                current_gray_frame, next_gray_frame,
                np.copy(init_params[i][0]),
                np.copy(init_params[i][1]))
            estimated_extrinsic_params.append([rmat, tvec])
            current_gray_frame = next_gray_frame

        capture.release()
        return estimated_extrinsic_params

    def transformPoints(self, object_points, rmat, tvec):
        transformed = np.matmul(rmat, object_points.T).T + tvec.T
        return transformed

    def calculate_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def calculate_error(self, object_points, params1, params2):
        transformed1 = [self.transformPoints(object_points, row[0], row[1])
                        for row in params1]
        transformed2 = [self.transformPoints(object_points, row[0], row[1])
                        for row in params2]
        error = [sum([self.calculate_distance(point1, point2)
                      for point1, point2 in zip(points1, points2)])
                 for points1, points2 in zip(transformed1, transformed2)]

        return error

    def show_error(self, error, algorithm_class):
        fig, ax = plt.subplots()
        ax.plot(range(len(error)), error, color='blue',
                label=str(algorithm_class))
        ax.plot(range(len(error)), error, 'o', color='blue')
        plt.legend()
        plt.show()

    def project_points(self, frame_size, object_points, camera_matrix, param):
        rmat = param[0]
        rvec, _ = cv2.Rodrigues(rmat)
        tvec = param[1]

        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, None)
        image_points = image_points.reshape((len(object_points), 2))
        # image_points += np.array([frame_size[0] // 2, frame_size[1] // 2])

        return image_points

    def visualize(self, video_path, frame_size, object_points,
                  camera_matrix, params1, params2):
        capture = cv2.VideoCapture(video_path)
        cv2.namedWindow("tracking")

        for param1, param2 in zip(params1, params2):
            success, frame = capture.read()

            image_points1 = self.project_points(
                frame_size, object_points, camera_matrix, param1)
            image_points2 = self.project_points(
                frame_size, object_points, camera_matrix, param2)

            cv2.polylines(frame, np.array([image_points1], dtype=np.int32),
                          True, (0, 255, 0))
            cv2.polylines(frame, np.array([image_points2], dtype=np.int32),
                          True, (0, 0, 255))

            while True:
                cv2.imshow('tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cv2.destroyAllWindows()
        capture.release()

    def run_test(self, test_path, algorithm_class):
        camera_matrix = load_matrix(test_path + '/camera_matrix.txt')
        width, height = get_screen_size(test_path + '/screen_parameters.csv')
        object_points = get_object_points(width, height)
        loaded_params = load_positions(test_path + '/positions.csv')
        extrinsic_params = format_params(loaded_params)
        frame_size = get_video_frame_size(test_path + '/video.mp4')

        algorithm_impl = algorithm_class(camera_matrix, object_points, frame_size)
        estimated_extrinsic_params = self.run_algorithm(
            algorithm_impl, test_path + '/video.mp4', extrinsic_params)

        error = self.calculate_error(
            object_points, extrinsic_params, estimated_extrinsic_params)
        self.show_error(error, algorithm_class)
        self.visualize(test_path + '/video.mp4',
                       frame_size,
                       object_points,
                       camera_matrix,
                       extrinsic_params, estimated_extrinsic_params)


if __name__ == '__main__':
    Testing().run_test('tests/living_room', FakeTracker)
