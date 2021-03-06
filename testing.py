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
    @staticmethod
    def run_algorithm(algorithm_impl, video_path, init_params, start_frame):
        capture = cv2.VideoCapture(video_path)
        success, frame = capture.read()
        for i in range(start_frame):
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

    @staticmethod
    def transform_points(object_points, rmat, tvec):
        transformed = np.matmul(rmat, object_points.T).T + tvec.T
        return transformed

    @staticmethod
    def calculate_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    @staticmethod
    def calculate_error(object_points, params1, params2):
        transformed1 = [Testing.transform_points(object_points, row[0], row[1])
                        for row in params1]
        transformed2 = [Testing.transform_points(object_points, row[0], row[1])
                        for row in params2]
        error = [sum([Testing.calculate_distance(point1, point2)
                      for point1, point2 in zip(points1, points2)])
                 for points1, points2 in zip(transformed1, transformed2)]

        return error

    @staticmethod
    def show_error(error, algorithm_class):
        fig, ax = plt.subplots()
        ax.plot(range(len(error)), error, color='blue',
                label=str(algorithm_class))
        ax.plot(range(len(error)), error, 'o', color='blue')
        plt.legend()
        plt.show()

    @staticmethod
    def project_points(frame_size, object_points, camera_matrix, param):
        rmat = param[0]
        rvec, _ = cv2.Rodrigues(rmat)
        tvec = param[1]

        image_points, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, None)
        image_points = image_points.reshape((len(object_points), 2))
        # image_points += np.array([frame_size[0] // 2, frame_size[1] // 2])

        return image_points

    @staticmethod
    def visualize(video_path, start_frame, frame_size, object_points,
                  camera_matrix, params1, params2):
        capture = cv2.VideoCapture(video_path)
        cv2.namedWindow("tracking")
        # writer = cv2.VideoWriter('out.avi',
        #                          cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'),
        #                          24,
        #                          frame_size)
        for i in range(start_frame):
            success, frame = capture.read()

        for param1, param2 in zip(params1, params2):
            success, frame = capture.read()

            image_points1 = Testing.project_points(
                frame_size, object_points, camera_matrix, param1)
            image_points2 = Testing.project_points(
                frame_size, object_points, camera_matrix, param2)

            cv2.polylines(frame, np.array([image_points1], dtype=np.int32),
                          True, (0, 255, 0))
            cv2.polylines(frame, np.array([image_points2], dtype=np.int32),
                          True, (0, 0, 255))
            # writer.write(frame)

            while True:
                cv2.imshow('tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # writer.release()
        cv2.destroyAllWindows()
        capture.release()

    @staticmethod
    def run_test(test_path, algorithm_class):
        camera_matrix = load_matrix(test_path + '/camera_matrix.txt')
        width, height = get_screen_size(test_path + '/screen_parameters.csv')
        object_points = get_object_points(width, height)
        loaded_params = load_positions(test_path + '/positions.csv')
        start_frame = 20
        end_frame = 22
        extrinsic_params = format_params(loaded_params)
        extrinsic_params = extrinsic_params[start_frame:end_frame]
        frame_size = get_video_frame_size(test_path + '/video.mp4')

        algorithm_impl = algorithm_class(camera_matrix, object_points, frame_size)
        estimated_extrinsic_params = Testing.run_algorithm(
            algorithm_impl, test_path + '/video.mp4', extrinsic_params,
            start_frame)

        error = Testing.calculate_error(
            object_points, extrinsic_params, estimated_extrinsic_params)
        Testing.show_error(error, algorithm_class)
        Testing.visualize(test_path + '/video.mp4',
                       start_frame,
                       frame_size,
                       object_points,
                       camera_matrix,
                       extrinsic_params, estimated_extrinsic_params)


if __name__ == '__main__':
    Testing().run_test('tests/living_room', FakeTracker)
