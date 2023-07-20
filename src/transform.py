import numpy as np
from pyquaternion import Quaternion


class Transform:
    def translate(vec, by):
        return vec + by

    def scale(vec, k):
        return vec * k

    def rotate(vec, by):
        if len(vec) == 2:
            c, s = np.cos(by), np.sin(by)
            R = np.array(((c, -s), (s, c)))
            rotated = np.matmul(R, vec)
            return np.array(rotated)
        else:
            rotated = Quaternion(by).rotate(Quaternion(0, vec[0], vec[1], vec[2]))
            return np.array([rotated.x, rotated.y, rotated.z])

    def similar(vec, params):
        return Transform.rotate(vec, params[0]) * params[1] + params[2]
