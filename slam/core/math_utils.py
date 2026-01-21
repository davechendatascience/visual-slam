import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_rot(qx, qy, qz, qw):
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def rot_to_quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return q[0], q[1], q[2], q[3]


def pose_inv(c2w):
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = -R.T @ t
    return w2c


def pose_compose(a, b):
    return a @ b

