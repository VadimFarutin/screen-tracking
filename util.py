import csv
import numpy as np
import cv2


def get_screen_size(screen_parameters_path):
    with open(screen_parameters_path, newline='') as csv_file:
        params_reader = csv.reader(csv_file, delimiter=',')
        size_parameters = next(params_reader)
        width = float(size_parameters[0])
        height = float(size_parameters[1])

    return width, height


def get_object_points(width, height):
    half_width = width / 2
    half_height = height / 2
    object_points = np.array([[-half_width, -half_height, 0.0],
                              [-half_width, half_height, 0.0],
                              [half_width, half_height, 0.0],
                              [half_width, -half_height, 0.0]],
                             dtype=np.float32)

    return object_points


def get_image_points(video_path):
    capture = cv2.VideoCapture(video_path)
    cv2.namedWindow("frame")

    image_points_all = []
    stopped = False

    while not stopped:
        success, frame = capture.read()
        image_points = []

        def on_mouse_click(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                image_points.append([x, y])
                cv2.circle(frame, (x, y), 6, (0, 0, 255), 1)

        cv2.setMouseCallback("frame", on_mouse_click)

        while True:
            cv2.imshow('frame', frame)
            if len(image_points) == 4:
                image_points_all.append(image_points)
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stopped = True
                break

    cv2.destroyAllWindows()
    capture.release()

    return np.array(image_points_all, dtype=np.float32)


def center_points(points_all, center):
    centered = [[point - [center[0] / 2, center[1] / 2] for point in points]
                for points in points_all]
    return np.array(centered, dtype=np.float32)


def get_video_frame_size(video_path):
    capture = cv2.VideoCapture(video_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    return width, height


def load_matrix(path):
    matrix = np.loadtxt(path)
    return matrix


def load_positions(positions_path):
    with open(positions_path, newline='') as csv_file:
        params_reader = csv.reader(csv_file, delimiter=',')
        extrinsic_params = [[float(x) for x in row] for row in params_reader]

    return extrinsic_params


def format_params(params):
    formatted = [[np.array([row[1:4],
                            row[4:7],
                            row[7:10]]),
                  np.array([[row[10]], [row[11]], [row[12]]])]
                 for row in params]
    return formatted


def project_points(points, rvec, tvec, camera_matrix):
    image_points, _ = cv2.projectPoints(
        np.array(points), rvec, tvec, camera_matrix, None)
    image_points = image_points.reshape((len(points), 2))
    return image_points


def project_points_int(points, rvec, tvec, camera_matrix):
    image_points = project_points(points, rvec, tvec, camera_matrix)
    return np.int32(np.rint(image_points))


def is_point_in(point, frame_size):
    return 0 <= point[0] < frame_size[0] and 0 <= point[1] < frame_size[1]


def rodrigues(src):
    dst, _ = cv2.Rodrigues(src)
    return dst
