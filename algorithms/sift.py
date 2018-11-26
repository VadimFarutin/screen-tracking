import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix, hog


class SiftTracker:
    OCTAVES_NUMBER = 4
    SAMPLES_PER_OCTAVE = 5
    SCALE_SPACE_SIGMA = 1 / math.sqrt(2)
    SCALE_SPACE_FACTOR = math.sqrt(2)
    CONTRAST_THRESHOLD = 1e9
    CORNER_THRESHOLD = 10

    def __init__(self, camera_mat, screen, frame_size):
        self.frame_size = frame_size

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):

        octaves = SiftTracker.generate_differences_of_gaussians(
            frame2_grayscale_mat)

        extrema = []

        for octave in octaves:
            height, width = octave[0].shape
            scale_x = self.frame_size[0] / width
            scale_y = self.frame_size[1] / height

            for i in range(1, len(octave) - 1):
                Hxx, Hxy, Hyy = hessian_matrix(octave[i], order='rc')
                histograms = SiftTracker.histograms(octave[i])

                for x in range(1, width - 1):
                    for y in range(1, height - 1):
                        hessian = [Hxx[y][x], Hxy[y][x], Hyy[y][x]]

                        is_keypoint = SiftTracker.is_keypoint(octave[i - 1],
                                                              octave[i],
                                                              octave[i + 1],
                                                              x, y,
                                                              hessian)

                        if is_keypoint:
                            histogram = histograms[y][x][0][0]
                            orientation = np.argmax(histogram)
                            extrema.append([x * scale_x, y * scale_y,
                                            scale_x, scale_y,
                                            orientation])

        # Keypoint localization, find subpixel extrema
        # Keypoint descriptor
        # Match features

        pos2_rotation_mat = pos1_rotation_mat
        pos2_translation = pos1_translation

        return pos2_rotation_mat, pos2_translation

    @staticmethod
    def L(image, sigma):
        return gaussian_filter(image, sigma)

    @staticmethod
    def generate_differences_of_gaussians(image):
        octaves = []
        initial_image = np.copy(image)

        for i in range(SiftTracker.OCTAVES_NUMBER):
            samples = []
            differences = []
            sigma = SiftTracker.SCALE_SPACE_SIGMA
            current_image = []

            for j in range(SiftTracker.SAMPLES_PER_OCTAVE):
                previous_image = np.copy(current_image)
                current_image = SiftTracker.L(initial_image, sigma)
                samples.append(current_image)
                sigma *= SiftTracker.SCALE_SPACE_FACTOR

                if len(previous_image) != 0:
                    differences.append(current_image - previous_image)
                    # while True:
                    #     cv2.imshow(str(sigma), differences[-1])
                    #     if cv2.waitKey(1) & 0xFF == ord('q'):
                    #         cv2.destroyAllWindows()
                    #         break

            octaves.append(differences)
            width, height = initial_image.shape
            initial_image = cv2.resize(initial_image, (height // 2, width // 2))

        return octaves

    @staticmethod
    def histograms(image):
        return hog(image,
                   orientations=36,
                   pixels_per_cell=(1, 1),
                   cells_per_block=(1, 1),
                   block_norm='L1',
                   feature_vector=False)

    @staticmethod
    def neighbours(lower, middle, higher, x, y):
        return np.concatenate((lower[y - 1:y + 2, x - 1:x + 2],
                               middle[y - 1:y + 2, x - 1:x + 2],
                               higher[y - 1:y + 2, x - 1:x + 2]),
                              axis=None)

    @staticmethod
    def is_extrema(lower, middle, higher, x, y):
        neighbours = SiftTracker.neighbours(lower, middle, higher, x, y)
        minima = min(neighbours)
        maxima = max(neighbours)
        point_value = middle[y][x]

        return point_value == minima or point_value == maxima

    @staticmethod
    def contrast_threshold_passed(image, x, y):
        return abs(image[y][x]) < SiftTracker.CONTRAST_THRESHOLD

    @staticmethod
    def is_corner(hxx, hxy, hyy):
        threshold_value = (SiftTracker.CORNER_THRESHOLD + 1) ** 2 \
                          / SiftTracker.CORNER_THRESHOLD
        trace = hxx + hyy
        det = hxx * hyy - hxy * hxy

        return det < 0 or trace ** 2 < threshold_value * det

    @staticmethod
    def is_keypoint(lower, middle, higher, x, y, hessian):
        hxx, hxy, hyy = hessian[0], hessian[1], hessian[2]

        is_extrema = SiftTracker.is_extrema(lower, middle, higher, x, y)
        is_contrast = SiftTracker.contrast_threshold_passed(middle, x, y)
        is_corner = SiftTracker.is_corner(hxx, hxy, hyy)

        return is_extrema and is_contrast and is_corner
