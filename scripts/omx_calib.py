"""
omx_calib.py - two-stage kinematic + hand-eye + mat-pose calibration for the OMX (Lab 9 Part D, assignment).

Parameter vector th (21):
    [0:5]   d1, l2z, l2x, l3, l4            link parameters (m)
    [5:9]   joint zero offsets o1..o4       (rad)   q_true = q_reported + o
    [9:15]  camera-mount correction          (rotvec[3], t[3]) applied after the starting mount T_EC0 (the measured one)
    [15:21] mat-pose correction T_BM         (rotvec[3], t[3])

Stage 1 (camera views) cannot observe l4, o4 (alias the camera mount), o1 (aliases mat yaw), l2x (aliases o2)
or mat z (aliases d1); FREE1 holds them at nominal. Stage 2 identifies l4 and o4 from tip touch points.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

NOMINAL = np.array([0.0975, 0.11315, 0.0415, 0.162, 0.123, 0, 0, 0, 0] + [0.0] * 12)
#                  d1 l2z l2x l3 l4  o1 o2 o3 o4  cam r(3) t(3)   mat r(3) x y z
FREE1 = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], bool)


def fk_np(q, th):
    d1, l2z, l2x, l3, l4 = th[:5]; q = np.asarray(q, float) + th[5:9]
    def Rz(t): c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    def Ry(t): c, s = np.cos(t), np.sin(t); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    R1 = Rz(q[0]); R2 = R1 @ Ry(q[1]); R3 = R2 @ Ry(q[2]); R4 = R3 @ Ry(q[3])
    t = np.array([0, 0, d1]) + R2 @ [l2x, 0, l2z] + R3 @ [l3, 0, 0] + R4 @ [l4, 0, 0]
    T = np.eye(4); T[:3, :3] = R4; T[:3, 3] = t; return T


def small_T(v):
    T = np.eye(4); T[:3, :3] = Rot.from_rotvec(v[:3]).as_matrix(); T[:3, 3] = v[3:]; return T


def residuals(x, th0, views, T_EC0, free=FREE1, w_rot=0.05):
    th = th0.copy(); th[free] = x
    T_ec = T_EC0 @ small_T(th[9:15]); T_bm = small_T(th[15:21]); out = []
    for q, T_cm in views:
        T_hat = np.linalg.inv(fk_np(q, th) @ T_ec) @ T_bm
        dT = np.linalg.inv(T_hat) @ T_cm
        out.append(dT[:3, 3]); out.append(w_rot * Rot.from_matrix(dT[:3, :3]).as_rotvec())
    return np.concatenate(out)


def fit_stage1(views, T_EC0, th0=NOMINAL, free=FREE1):
    """views: list of (q_reported (4,), T_CM (4x4)). Returns (th, result)."""
    sol = least_squares(residuals, th0[free], args=(th0, views, T_EC0, free), x_scale='jac')
    th = th0.copy(); th[free] = sol.x
    sv = np.linalg.svd(sol.jac, compute_uv=False)
    return th, dict(rms_mm=1e3 * np.sqrt(np.mean(residuals(sol.x, th0, views, T_EC0, free).reshape(-1, 6)[:, :3] ** 2)),
                    cond=sv[0] / sv[-1], n=len(views))


def pose_errors(th, views, T_EC0):
    r = residuals(th[FREE1], th, views, T_EC0).reshape(-1, 6)
    e = 1e3 * np.linalg.norm(r[:, :3], axis=1); return e.mean(), e.max()


def fit_touch(th, touches):
    """touches: list of (q_reported, p_tip_in_base (3,)). Identifies l4 and o4. Returns th_final."""
    def res(x):
        t = th.copy(); t[4] = x[0]; t[8] = x[1]
        return np.concatenate([fk_np(q, t)[:3, 3] - p for q, p in touches])
    s = least_squares(res, [th[4], th[8]])
    out = th.copy(); out[4], out[8] = s.x
    return out, 1e3 * np.abs(res(s.x)).reshape(-1, 3)
