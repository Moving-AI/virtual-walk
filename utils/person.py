import cv2
import numpy as np
import logging


FORMAT = "%(asctime)s - %(levelname)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger(__name__)

formatter = logging.Formatter(FORMAT)
logger.setLevel(logging.INFO)

PARTS = {
    0: 'NOSE',
    1: 'LEFT_EYE',
    2: 'RIGHT_EYE',
    3: 'LEFT_EAR',
    4: 'RIGHT_EAR',
    5: 'LEFT_SHOULDER',
    6: 'RIGHT_SHOULDER',
    7: 'LEFT_ELBOW',
    8: 'RIGHT_ELBOW',
    9: 'LEFT_WRIST',
    10: 'RIGHT_WRIST',
    11: 'LEFT_HIP',
    12: 'RIGHT_HIP',
    13: 'LEFT_KNEE',
    14: 'RIGHT_KNEE',
    15: 'LEFT_ANKLE',
    16: 'RIGHT_ANKLE',
    17: 'NECK',
    18: 'HIP'
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Person:
    def __init__(self, heatmap=None, offsets=None, rescale=(1,1), threshold=0.7, path_txt=None, show_head=False):
        if path_txt is None:
            self.keypoints = self.get_keypoints(heatmap, offsets)
            self.keypoints.append(self._infer_neck())
            self.threshold = threshold
            # self.keypoints.append(self._infer_hip())
            self.pose = self.infer_pose(self.keypoints) # This is useless for now TODO:CHANGE

            self.inferred_points = [list(range(19))]
        else:
            self.keypoints = self.skeleton_from_txt(path_txt)
        # LIMBS
        self.pairs = [
            (5, 6),
            (5, 7),
            (7, 9),
            (5, 11),
            (11, 13),
            (13, 15),
            (6, 8),
            (8, 10),
            (6, 12),
            (12, 14),
            (14, 16),
            (11, 12)
        ]
        self.rescale = rescale
        self.threshold = threshold
        self.H = self.get_height()
        self.W = self.get_width()

        if rescale[0] != 1. or rescale[1] != 1.:
            self.get_coords = self._get_coords_rescaled
            self.get_limbs = self._get_limbs_rescaled
            self.rescale = rescale
        else:
            self.get_coords = self._get_coords
            self.get_limbs = self._get_limbs

        self.keypoints_positions = self.get_keypoints_array()

    def get_keypoints(self, heatmaps, offsets, output_stride=32):
        scores = sigmoid(heatmaps)
        num_keypoints = scores.shape[2]
        heatmap_positions = []
        offset_vectors = []
        confidences = []
        for ki in range(0, num_keypoints):
            x, y = np.unravel_index(np.argmax(scores[:, :, ki]), scores[:, :, ki].shape)
            confidences.append(scores[x, y, ki])
            offset_vector = (offsets[x, y, ki], offsets[x, y, num_keypoints + ki])
            heatmap_positions.append((x, y))
            offset_vectors.append(offset_vector)
        image_positions = np.add(np.array(heatmap_positions) * output_stride, offset_vectors)
        keypoints = [KeyPoint(i, np.flip(pos), confidences[i]) for i, pos in enumerate(image_positions)]

        return keypoints

    def infer_pose(self, coords):
        return "Unknown"

    def _get_coords(self):
        return [kp.point() for kp in self.keypoints if kp.confidence > self.threshold]

    def _get_coords_rescaled(self):
        return [kp.point_rescaled(self.rescale) for kp in self.keypoints if kp.confidence > self.threshold]

    def _get_limbs(self):
        limbs = [(self.keypoints[i].point(), self.keypoints[j].point()) for i, j in self.pairs if
                 self.keypoints[i].confidence > self.threshold and self.keypoints[j].confidence > self.threshold]
        return list(filter(lambda x: x is not None, limbs))

    def _get_limbs_rescaled(self):
        limbs = [(self.keypoints[i].point_rescaled(self.rescale), self.keypoints[j].point_rescaled(self.rescale)) for
                 i, j in self.pairs if
                 self.keypoints[i].confidence > self.threshold and self.keypoints[j].confidence > self.threshold]
        return list(filter(lambda x: x is not None, limbs))

    def confidence(self):
        return np.mean([k.confidence for k in self.keypoints])

    def to_string(self):
        return "\n".join([a.to_string() for a in self.keypoints])

    def draw_points(self, img):
        radius = 1
        color = (0, 0, 255)  # BGR
        thickness = 3
        for p in self.get_coords():
            cv2.circle(img, p, radius, color, thickness)
        for p1, p2 in self.get_limbs():
            cv2.line(img, p1, p2, color, thickness)
        return img

    def skeleton_to_txt(self, path):
        with open(path, 'w') as F:
            for i, k in enumerate(self.keypoints):
                F.write(str(i) + '\t' + str(k.x) + '\t' + str(k.y) + '\t' + str(k.confidence) + '\n')

    @staticmethod
    def skeleton_from_txt(path):
        keypoints = []
        with open(path, 'r') as F:
            for line in F.readlines():
                i, kx, ky, conf = [float(c) for c in line.split('\t')]
                keypoints.append(KeyPoint(int(i), (kx, ky), conf))
        return keypoints

    def _infer_neck(self):
        lshoulder = [kp for kp in self.keypoints if kp.body_part == 'LEFT_SHOULDER'][0]
        rshoulder = [kp for kp in self.keypoints if kp.body_part == 'RIGHT_SHOULDER'][0]
        neckx, necky = (rshoulder.x + lshoulder.x) / 2, (rshoulder.y + lshoulder.y) / 2
        confidence = min(lshoulder.confidence, rshoulder.confidence)
        neck = KeyPoint(17, (neckx, necky), confidence)
        return neck

    def _infer_hip(self):
        lhip = [kp for kp in self.keypoints if kp.body_part == 'LEFT_HIP'][0]
        rhip = [kp for kp in self.keypoints if kp.body_part == 'RIGHT_HIP'][0]
        hipx, hipy = (lhip.x + rhip.x) / 2, (lhip.y + rhip.y) / 2
        confidence = min(lhip.confidence, rhip.confidence)
        hip = KeyPoint(18, (hipx, hipy), confidence)
        return hip

    def get_width(self):
        '''
        keypoints: 5: LEFT SHOULDER, 6: RIGHT SHOULDER.
        :return:
        '''
        lshoulder_x = self.keypoints[5].x  if self.keypoints[5].confidence > self.threshold else None
        rshoulder_x = self.keypoints[6].x  if self.keypoints[6].confidence > self.threshold else None

        if lshoulder_x is not None and rshoulder_x is not None:
            return abs(lshoulder_x - rshoulder_x)
        else:
            return 0

    def get_height(self):
        '''
        keypoints: 15: LEFT FOOT, 16: RIGHT FOOT, 0: NOSE.
        :return:
        '''
        cand_inf = [kp for kp in self.keypoints[11:13] if kp.confidence > self.threshold]
        cand_sup = [kp for kp in self.keypoints[0:5] if kp.confidence > self.threshold]

        if len(cand_inf) > 0 and len(cand_sup) > 0:
            cand_inf_y = sorted(cand_inf, key=lambda x: -x.y)[0].y
            cand_sup_y = sorted(cand_sup, key=lambda x: x.y)[0].y
            return cand_inf_y - cand_sup_y
        else:
            return 0

    def infer_lc_keypoints(self, prev_person):
        """This function will be used when creating the frame groups.
        It takes the Person of the previous frame and uses its keypoits
        to infer the keypoints of "self".
        If the keypoint has low confidence, the position of the keypoint
        relative to the neck is used in the current person

        
        Args:
            prev_person (Person): The person from the current frame
        """
        [self.infer_point(index, prev_person) for index, kp in enumerate(self.keypoints)\
            if self.keypoints[index].confidence < self.threshold and index in self.inferred_points]
        self.get_height()
        self.get_width()
        

    def infer_point(self, index, prev_person):
        """Use the position of the neck and the same keypoint from the previous frame to infer this one.
        Same confidence is applied

        Args:
            index (int): Index of the keypoint to infer
            prev_person (Person): Person from the previous frame
        """
        # Use the position of the neck and the same keypoint from the previous frame to infer this one. Same confidence
        # is applied
        
        if self.keypoints[index].confidence < self.threshold:
            xi = self.keypoints[17].x + prev_person.keypoints[index].x - prev_person.keypoints[17].x
            yi = self.keypoints[17].y + prev_person.keypoints[index].y - prev_person.keypoints[17].y
            self.keypoints[index] = KeyPoint(index, (xi, yi), prev_person.keypoints[index].confidence)

    def get_keypoints_array(self):
        return np.array([(kp.x, kp.y) for kp in self.keypoints])

    def low_confidence_keypoints(self):
        return np.array([kp.index for kp in self.keypoints if kp.confidence > self.threshold])

    def is_valid_first(self):
        """This function determines if the frame should be considered for training.
        Before it was embedded inside the pipeline of DataProcessor. Now it's a function,
        so conditions can be changed based on performance.
        
        Returns:
            bool: True if is valid
        """
        return self.H > 0 and self.W > 0
    
    def is_valid_other(self):

        return True


class KeyPoint:
    def __init__(self, index, pos, v):
        x, y = pos
        self.x = x
        self.y = y
        self.index = index
        self.body_part = PARTS.get(index)
        self.confidence = v

    def point(self):
        return int(self.y), int(self.x)

    def point_rescaled(self, rescale):
        return int(self.y * rescale[0]), int(self.x * rescale[1])

    def to_string(self):
        return 'part: {} location: {} confidence: {}'.format(self.body_part, (self.x, self.y), self.confidence)
