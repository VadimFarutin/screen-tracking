import cv2
import numpy as np
import math as math

from util import project_points_int


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

    def __init__(self, camera_mat, object_points, frame_size):
        self.camera_mat = camera_mat
        self.object_points = object_points
        self.frame_size = np.array(list(frame_size))
        self.vecSpeed = np.array([[0], [0], [0], [0], [0], [0]],
                                 dtype=np.float32)

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        controlPoints, controlPointsPair = self.control_points(
            self.object_points, RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER)
        controlPoints = [controlPoints[i * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER:
                                       (i + 1) * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                         for i in range(RapidScreenTracker.EDGE_NUMBER)]
        controlPointsPair = [controlPointsPair[i * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER:
                                               (i + 1) * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                             for i in range(RapidScreenTracker.EDGE_NUMBER)]

        imagePoints = []
        imagePointsIdx = []
        edgeLines = []
        rejectedPoints = []

        pos1_rotation, _ = cv2.Rodrigues(pos1_rotation_mat)
        shift = self.frame_size // 2
        frame2_gradient_map = cv2.Laplacian(frame2_grayscale_mat, cv2.CV_64F)

        for i in range(RapidScreenTracker.EDGE_NUMBER):
            R = controlPoints[i]
            S = controlPointsPair[i]
            r = project_points_int(
                R, pos1_rotation, pos1_translation, self.camera_mat)
            s = project_points_int(
                S, pos1_rotation, pos1_translation, self.camera_mat)
            # r = RapidScreenTracker.change_coordinate_system(r, shift)
            # s = RapidScreenTracker.change_coordinate_system(s, shift)

            foundPoints, foundPointsIdx = self.search_edge(
                r, s, frame2_gradient_map, i)
            # foundPoints = RapidScreenTracker.change_coordinate_system(foundPoints, -shift)

            if len(foundPoints) == 0:
                continue

            corners = RapidScreenTracker.linear_regression(foundPoints)
            edgeLines.append(corners)
            error = RapidScreenTracker.find_edge_error(foundPoints, corners)

            if error > RapidScreenTracker.EDGE_ERROR_THRESHOLD:
                continue

            accepted, acceptedIdx, rejected = RapidScreenTracker.filter_edge_points(
                foundPoints, foundPointsIdx, corners)
            imagePoints.extend(accepted)
            imagePointsIdx.extend(acceptedIdx)
            rejectedPoints.extend(rejected)

        imagePoints = np.array(imagePoints, np.float32)
        allControlPoints = np.array([controlPoints[i // RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                                                  [i % RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER]
                                     for i in imagePointsIdx])

        lastRVec = np.copy(pos1_rotation)
        lastTVec = np.copy(pos1_translation)

        if len(imagePoints) < 3:
            rvec = np.copy(pos1_rotation)
            tvec = np.copy(pos1_translation)
        else:
            _, rvec, tvec = cv2.solvePnP(allControlPoints, imagePoints, self.camera_mat, None,
                                         pos1_rotation, pos1_translation, useExtrinsicGuess=True)

        # retval, rvec, tvec, _ = cv2.solvePnPRansac(
        #     allControlPoints, imagePoints, cameraMatrix, None)

        diffRVec = rvec - lastRVec
        diffTVec = tvec - lastTVec

        rvec = lastRVec + RapidScreenTracker.ALPHA * diffRVec \
            + self.vecSpeed[0:3]
        tvec = lastTVec + RapidScreenTracker.ALPHA * diffTVec \
            + self.vecSpeed[3:6]
        self.vecSpeed += RapidScreenTracker.BETA \
            * np.append(rvec - lastRVec, tvec - lastTVec, axis=0)

        rmat, _ = cv2.Rodrigues(rvec)
        return rmat, tvec

    @staticmethod
    def change_coordinate_system(points, shift):
        return points + shift

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

    def search_edge(self, r, s, gradientMap, edgeIndex):
        cos_a = np.array([(si[0] - ri[0]) / np.linalg.norm(si - ri)
                          for si, ri in zip(s, r)])
        sin_a = np.array([(si[1] - ri[1]) / np.linalg.norm(si - ri)
                          for si, ri in zip(s, r)])
        tana = sin_a / cos_a

        foundPoints = []
        foundPointsIdx = []

        for j in range(RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER):
            step_x, step_y = RapidScreenTracker.get_search_direction(tana[j])

            if not 0 <= r[j][0] < self.frame_size[0] \
                    or not 0 <= r[j][1] < self.frame_size[1]:
                continue

            point = self.search_edge_from_point(
                gradientMap, r[j], (step_x, step_y))
            foundPoints.append(point)
            foundPointsIdx.append(
                edgeIndex * RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER + j)

        return foundPoints, foundPointsIdx

    def search_edge_from_point(self, edges, start, step):
        maxGradientPoint = np.copy(start)
        maxGradient = abs(edges[start[1], start[0]])

        maxGradient, maxGradientPoint = \
            self.search_edge_from_point_to_one_side(
                edges, start, step, RapidScreenTracker.NUMBER_OF_STEPS,
                maxGradient, maxGradientPoint)
        step = (-step[0], -step[1])
        maxGradient, maxGradientPoint = \
            self.search_edge_from_point_to_one_side(
                edges, start, step, RapidScreenTracker.NUMBER_OF_STEPS,
                maxGradient, maxGradientPoint)

        return maxGradientPoint

    def search_edge_from_point_to_one_side(
            self, edges, start, step, count, maxGradient, maxGradientPoint):
        current = np.copy(start)

        for i in range(count):
            current += step

            if not 0 <= current[0] < self.frame_size[1] \
                    or not 0 <= current[1] < self.frame_size[0]:
                continue

            gradientValue = abs(edges[current[1], current[0]])

            if maxGradient < gradientValue:
                maxGradient = gradientValue
                maxGradientPoint = np.copy(current)

        return maxGradient, maxGradientPoint

    @staticmethod
    def linear_regression(points):
        x = [point[0] for point in points]
        y = [point[1] for point in points]

        if abs(x[0] - x[-1]) >= abs(y[0] - y[-1]):
            A = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            corners = np.array([[x[0], k * x[0] + b],
                                [x[-1], k * x[-1] + b]])
        else:
            A = np.vstack([y, np.ones(len(y))]).T
            k, b = np.linalg.lstsq(A, x, rcond=None)[0]
            corners = np.array([[k * y[0] + b, y[0]],
                                [k * y[-1] + b, y[-1]]])

        return corners

    @staticmethod
    def find_edge_error(foundPoints, corners):
        error = 0

        for point in foundPoints:
            error += np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[0] - point)) \
                / np.linalg.norm(corners[1] - corners[0])
        error /= RapidScreenTracker.EDGE_CONTROL_POINTS_NUMBER

        return error

    @staticmethod
    def filter_edge_points(foundPoints, foundPointsIdx, corners):
        accepted = []
        acceptedIdx = []
        rejected = []

        for point, j in zip(foundPoints, foundPointsIdx):
            d = np.linalg.norm(
                np.cross(corners[1] - corners[0], corners[0] - point)) \
                / np.linalg.norm(corners[1] - corners[0])
            if d <= RapidScreenTracker.EDGE_POINT_FILTER_THRESHOLD:
                accepted.append(point)
                acceptedIdx.append(j)
            else:
                rejected.append(point)

        return accepted, acceptedIdx, rejected

    @staticmethod
    def control_points(objectPoints, oneSideCount):
        points = np.copy(objectPoints)
        points = np.append(points, [objectPoints[0]], axis=0)

        controlPoints = [list(point * (j + 1) / (oneSideCount + 1)
                         + nextPoint * (oneSideCount - j) / (oneSideCount + 1))
                         for (point, nextPoint) in zip(points[:-1], points[1:])
                         for j in range(oneSideCount)]
        controlPointPairs = [point
                             for point in points[:-1]
                             for j in range(oneSideCount)]

        return controlPoints, controlPointPairs
