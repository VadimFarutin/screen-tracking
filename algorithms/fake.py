class FakeTracker:
    def __init__(self, camera_mat, screen, frame_size):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):

        pos2_rotation_mat = pos1_rotation_mat
        pos2_translation = pos1_translation

        return pos2_rotation_mat, pos2_translation
