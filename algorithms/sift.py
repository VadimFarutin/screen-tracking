import math
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


class SiftTracker:
    def __init__(self, camera_mat, screen, frame_size):
        pass

    def draw(self):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):

        # Generate features
        ## Scale-space extrema detection
        differences, filtered_image = self.D_stack(frame2_grayscale_mat, 1.6)

        # while True:
        #     cv2.imshow('filtered', difference)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        ## Keypoint localization
        ## Orientation assignment
        ## Keypoint descriptor

        # Match features

        pos2_rotation_mat = pos1_rotation_mat
        pos2_translation = pos1_translation

        return pos2_rotation_mat, pos2_translation

    def L(self, image, sigma):
        return gaussian_filter(image, sigma)

    def D(self, image, sigma):
        k = math.sqrt(2)
        return self.L(image, k * sigma) - self.L(image, sigma)

    def D_stack(self, image, sigma):
        current = np.copy(image)
        filtered = self.L(image, sigma)
        stack = []

        for i in range(5):
            current = np.copy(filtered)
            filtered = self.L(current, sigma)
            stack.append(filtered - current)

        return stack, current - stack[-1]
