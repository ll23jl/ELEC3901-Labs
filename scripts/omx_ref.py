"""Reference implementations of the ELEC3901 lab task functions (generated from tasks.py).
Each lab imports the functions of PREVIOUS labs from here so nobody is blocked by last week; your own definition in
the current notebook takes precedence. Do not read ahead: the point of the tasks is to write them yourself."""
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.transform import Rotation as _Rot
NOM = {'d1': 0.0975, 'l2z': 0.11315, 'l2x': 0.0415, 'l3': 0.162, 'l4': 0.123}
Q_LIM = np.array([[-2.618, 2.618], [-1.868, 1.658], [-1.571, 1.676], [-1.745, 1.745]])
def small_T(v):
    T = np.eye(4); T[:3, :3] = _Rot.from_rotvec(v[:3]).as_matrix(); T[:3, 3] = v[3:]; return T
