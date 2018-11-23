
class RapidScreenTracker:
    def __init__(self, camera_mat, screen):
        pass

    def track(self, frame1_grayscale_mat, frame2_grayscale_mat,
              pos1_rotation_mat, pos1_translation):
        # todo
        pos2_rotation_mat = pos1_rotation_mat
        pos2_translation = pos1_translation

        return pos2_rotation_mat, pos2_translation

    def draw(self):
        pass
