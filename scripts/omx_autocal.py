"""
omx_autocal.py - automatic wrist-camera calibration: the ARM moves the camera over the mat, waits until it is static at each
view, grabs the frame, and fits the intrinsics (K, lens distortion) from the mat markers. It also measures the camera mount.

    python omx_autocal.py --arm-id OMX-07 --show               # real arm (torque ON: keep the workspace clear)
    python omx_autocal.py --sim                                # digital twin: compares the result with the true K

    from omx_autocal import auto_calibrate                    # from a notebook, after the connect cell
    K, dist, rms = auto_calibrate(arm, cam, ARM_ID, show=True)

Two stages, so that nothing depends on the camera mount (T_EC in omx_model is a CAD placeholder: working out the real one is the
students' job; the first real mount was measured at 58.6 deg below the gripper axis, 29 Sep 2026):
  1. bootstrap - 20 poses aimed at the mat around x = 200 mm with a GUESSED mount (MOUNT_PRIOR_DEG = 60, chosen to work for
     mounts of 45-70 deg; with fewer than 6 good views it falls back to a mount-independent pitch sweep). Frames showing >= 5
     markers give a first K and distortion, and then the camera mount T_EC from each view's mat pose (mat frame = base frame,
     nominal kinematics).
  2. planned views - calibration_views() with the MEASURED mount and camera: reachable within the joint limits, tip >= 50 mm
     above the mat, >= 8 markers fully in view, diverse in camera height, tilt and azimuth.
The final fit uses every frame. fx = fy is imposed (square pixels): with the wrist roll locked, every view is tilted about
the same camera axis, which on its own cannot separate fy from cy; the principal point and k1, k2 stay free (the first real
camera, an Innomaker U20CAM-720P, has its principal point ~50 px off centre and k1 ~ -0.46).

Every frame is taken with the joints settled (< 0.002 rad), stale frames drained, and two consecutive images agreeing
(median corner movement < still_px). Saved: camera_K_session_<ARM_ID>.npz (as the hand-held routine) and
camera_autocal_<ARM_ID>.npz (K, dist, the measured T_EC, and the joint angles and detections of every view - the kind of data
the Lab 9 kinematic / hand-eye calibration uses).
"""
import argparse, os, time
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as Rot

from omx_model import omx_ik, T_EC, Q_LIM, HOME_Q
from omx_calib import fk_np, NOMINAL
from omx_camera import load_mat

K_NOMINAL = np.array([[950.0, 0, 640], [0, 950.0, 360], [0, 0, 1]])
SIZE = (1280, 720)
HOME = HOME_Q
# Stage 1 aims the camera with a GUESSED mount (the mount is then measured, never taken from this): the first arm measured
# 58.6-62.0 deg below the gripper axis, camera ~(-24, 0, 20) mm in {E} (29-30 Sep 2026). A mount far from it falls back to sweep_views.
MOUNT_PRIOR_DEG = 60.0
MOUNT_PRIOR_T = (-0.024, 0.0, 0.020)
FIT_FLAGS = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_USE_INTRINSIC_GUESS


def Rx(a):
    """Rotation about x_E by the wrist roll a (rad). The roll axis lies along link 4 (x_E), through the EE point, and the
    camera is fixed to the rolling gripper base, so T_BC = T_BE(q) Rx(roll) T_EC."""
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def Ry(a):
    """Rotation about y (rad), used to nudge the camera when planning views."""
    c, s = np.cos(a), np.sin(a)
    T = np.eye(4); T[0, 0] = T[2, 2] = c; T[0, 2] = s; T[2, 0] = -s
    return T


def fk_view(v):
    """T_BE for a view: 4 joint angles, or 4 + roll (rad)."""
    v = np.asarray(v, float)
    T = fk_np(v[:4], NOMINAL)
    return T @ Rx(v[4]) if len(v) > 4 else T


def T_EC_nominal():
    return np.asarray(T_EC.A)


def visible_markers(q, K=K_NOMINAL, size=SIZE, margin=20, corners_M=None, min_px=18, max_incidence_deg=65, dist=None,
                    T_ec=None):
    """Ids of the mat markers fully inside the image (with margin), big enough and not too oblique, at joint angles q
    (nominal kinematics, mat frame = base frame, camera mount T_ec - default the CAD placeholder)."""
    corners_M = load_mat(scale=1.0)[1] if corners_M is None else corners_M
    T_BC = fk_np(q, NOMINAL) @ (T_EC_nominal() if T_ec is None else T_ec)
    T_CB = np.linalg.inv(T_BC)
    if T_BC[2, 2] > -0.2:                                             # optical axis must point down at the mat
        return []
    rvec = cv2.Rodrigues(T_CB[:3, :3])[0]; tvec = T_CB[:3, 3]
    ids = []
    for i, P in corners_M.items():
        Pc = (T_CB[:3, :3] @ P.T).T + T_CB[:3, 3]
        if np.any(Pc[:, 2] < 0.05):
            continue
        uv = cv2.projectPoints(P.astype(np.float64), rvec, tvec, K, np.zeros(5) if dist is None else dist)[0].reshape(-1, 2)
        if np.any(uv < margin) or np.any(uv[:, 0] > size[0] - margin) or np.any(uv[:, 1] > size[1] - margin):
            continue
        if np.min(np.linalg.norm(np.roll(uv, 1, 0) - uv, axis=1)) < min_px:
            continue
        view = Pc.mean(0) / np.linalg.norm(Pc.mean(0))                  # ray to the marker centre
        n_C = T_CB[:3, :3] @ np.array([0, 0, 1.0])                      # mat normal in the camera frame
        if np.degrees(np.arccos(abs(view @ n_C))) > max_incidence_deg:
            continue
        ids.append(i)
    return ids


def mount_prior(tilt_deg=None):
    """A camera mount with the optical axis tilt_deg below the gripper axis at MOUNT_PRIOR_T: only a guess, to aim stage 1."""
    a = np.radians(MOUNT_PRIOR_DEG if tilt_deg is None else tilt_deg); T = np.eye(4)
    T[:3, :3] = [[0, -np.sin(a), np.cos(a)], [-1, 0, 0], [0, -np.cos(a), -np.sin(a)]]; T[:3, 3] = MOUNT_PRIOR_T
    return T


def bootstrap_views(min_tip_z=0.08, rolls_deg=None, n=20, mount_deg=MOUNT_PRIOR_DEG, mount_spread_deg=10, min_markers=6,
                    K=K_NOMINAL):
    """Stage-1 poses, aimed at the mat around x = 200 mm (aim points at x 170-265 mm, |y| < 80 mm) for a mount guessed at
    mount_deg: tool-tip heights 0.12-0.20 m, radii 0.13-0.25 m, base turned up to +/-17 deg, the tool pitch solved so the
    optical axis meets the mat 0.18-0.26 m out. A pose is kept only if it sees >= min_markers for every mount in
    mount_deg +/- mount_spread_deg (and every roll in rolls_deg); n are chosen for diversity in camera height, tilt and azimuth.
    Checked: all 20 see >= 5 markers for true mounts of 45-70 deg (f 1030 px). The old pitch sweep looked back at the base
    with a 60 deg mount (first arm, 30 Sep 2026); mount_deg=None still gives it (sweep_views), the fallback for a mount far
    from the guess."""
    if mount_deg is None:
        return sweep_views(min_tip_z, rolls_deg)
    corners_M = load_mat(scale=1.0)[1]
    mounts = [mount_prior(m) for m in (mount_deg - mount_spread_deg, mount_deg, mount_deg + mount_spread_deg)]
    if rolls_deg:
        mounts += [Rx(np.radians(r)) @ mounts[1] for r in rolls_deg if r]
    T_aim = mounts[1]; cands = []
    for az in np.radians((-17, -11, -6, 0, 6, 11, 17)):
        for rho in (0.18, 0.21, 0.24, 0.26):
            for r in (0.13, 0.17, 0.21, 0.25):
                for zt in (0.12, 0.16, 0.20):
                    best = None
                    for pitch in np.radians(np.arange(-10, 101, 2)):
                        q = omx_ik([r * np.cos(az), r * np.sin(az), zt], pitch, elbow='any')
                        if q is None or fk_np(q, NOMINAL)[2, 3] < min_tip_z:
                            continue
                        T_BC = fk_np(q, NOMINAL) @ T_aim; c, z = T_BC[:3, 3], T_BC[:3, 2]
                        if z[2] > -0.2:
                            continue
                        e = abs(np.hypot(*(c - z * c[2] / z[2])[:2]) - rho)        # where the optical axis meets the mat
                        if best is None or e < best[0]:
                            best = (e, q, T_BC)
                    if best is None or best[0] > 0.01:
                        continue
                    q, T_BC = best[1], best[2]
                    if all(len(visible_markers(q, K, corners_M=corners_M, T_ec=T)) >= min_markers for T in mounts):
                        tilt = np.degrees(np.arccos(np.clip(-T_BC[2, 2], -1, 1)))
                        cands.append(dict(q=q, h=T_BC[2, 3], tilt=tilt, dirn=T_BC[:2, 2], az=np.degrees(az)))
    if len(cands) < n:
        raise RuntimeError(f'stage 1: only {len(cands)} aimed poses found for a {mount_deg} deg mount')
    feat = np.array([[c['h'] / 0.03, c['tilt'] / 8 * c['dirn'][0] / (np.linalg.norm(c['dirn']) + 1e-9),
                      c['tilt'] / 8 * c['dirn'][1] / (np.linalg.norm(c['dirn']) + 1e-9), c['az'] / 8] for c in cands])
    chosen = [int(np.argmax([c['tilt'] for c in cands]))]
    d = np.linalg.norm(feat - feat[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d)); chosen.append(k); d = np.minimum(d, np.linalg.norm(feat - feat[k], axis=1))
    Q = [cands[k]['q'] for k in chosen]; out, last = [], np.asarray(HOME)       # short moves: nearest next pose, from home
    while Q:
        k = int(np.argmin([np.abs(q - last).max() for q in Q])); last = Q.pop(k); out.append(last)
    out = np.array(out)
    if rolls_deg:                                                 # cycle the roll through the list: one extra column (rad)
        r = np.radians([rolls_deg[k % len(rolls_deg)] for k in range(len(out))])
        out = np.column_stack([out, r])
    return out


def sweep_views(min_tip_z=0.08, rolls_deg=None):
    """Fallback stage-1 poses that see the mat whatever the camera mount: tip 0.18/0.22 m out and 0.14 m up, base turned
    +/-11 deg, tool pitched 0-80 deg below horizontal (low pitches suit steeply tilted mounts, high ones a camera looking
    along the tool). Checked: >= 10 of the 20 see >= 5 markers for mounts of 0, 30 and 60 deg - but with a 60 deg mount the
    steep pitches look back at the base."""
    out = []
    for az in (-0.2, 0.2):
        for r in (0.18, 0.22):
            for pitch in ((0, 20, 40, 60, 80) if (r == 0.18) == (az < 0) else (80, 60, 40, 20, 0)):   # short moves
                q = omx_ik([r * np.cos(az), r * np.sin(az), 0.14], np.radians(pitch), elbow='any')
                if q is not None and fk_np(q, NOMINAL)[2, 3] >= min_tip_z:
                    out.append(q)
    out = np.array(out)
    if rolls_deg:                                                 # cycle the roll through the list: one extra column (rad)
        r = np.radians([rolls_deg[k % len(rolls_deg)] for k in range(len(out))])
        out = np.column_stack([out, r])
    return out


def calibration_views(n=18, K=K_NOMINAL, size=SIZE, min_markers=8, min_tip_z=0.05, dist=None, T_ec=None, pose_margin_deg=4.0):
    """-> (n, 4) joint configurations for the calibration, diverse in camera height, tilt and azimuth, ordered for short moves.
    Candidates: tool-tip heights, radii and azimuths over the mat and tool pitches 0-125 deg below horizontal; each is checked
    with the given camera (K, dist) and mount T_ec (default: the CAD placeholder). The nominal kinematics are not exact, so a
    view must still see min_markers with the camera turned pose_margin_deg either way about its x and y axes (first arm,
    30 Sep 2026: with the measured mount, the view on the mat drifted up to 18 mm with q4 and 6 of 18 planned views saw only
    4-7 markers; a 4 deg margin under-predicts the count on 30 of that run's 31 views)."""
    corners_M = load_mat(scale=1.0)[1]
    T_ec = T_EC_nominal() if T_ec is None else T_ec
    a = np.radians(pose_margin_deg)
    T_nudged = [T_ec @ Rx(s * a) for s in (1, -1)] + [T_ec @ Ry(s * a) for s in (1, -1)] if a else []
    cands = []
    for z in (0.07, 0.10, 0.13, 0.16, 0.19):
        for r in (0.14, 0.17, 0.20, 0.23, 0.26):
            for az in np.radians(np.arange(-50, 51, 10)):
                for pitch in np.radians((0, 15, 30, 45, 60, 75, 90, 105, 120)):
                    q = omx_ik([r * np.cos(az), r * np.sin(az), z], pitch, elbow='any')
                    if q is None or z < min_tip_z:
                        continue
                    ids = visible_markers(q, K, size, corners_M=corners_M, dist=dist, T_ec=T_ec)
                    if len(ids) >= min_markers and all(
                            len(visible_markers(q, K, size, corners_M=corners_M, dist=dist, T_ec=Tn)) >= min_markers for Tn in T_nudged):
                        T_BC = fk_np(q, NOMINAL) @ T_ec
                        tilt = np.degrees(np.arccos(np.clip(-T_BC[2, 2], -1, 1)))
                        if tilt > 50:
                            continue
                        cands.append(dict(q=q, h=T_BC[2, 3], tilt=tilt, az=np.degrees(q[0]), dirn=T_BC[:2, 2], n=len(ids)))
    if len(cands) < n:
        raise RuntimeError(f'only {len(cands)} usable calibration views found (check the camera mount / camera model)')
    # greedy farthest-point selection in (height, tilt direction, azimuth): start from the most tilted view
    feat = np.array([[c['h'] / 0.04, c['tilt'] / 15 * c['dirn'][0] / (np.linalg.norm(c['dirn']) + 1e-9),
                      c['tilt'] / 15 * c['dirn'][1] / (np.linalg.norm(c['dirn']) + 1e-9), c['az'] / 25] for c in cands])
    chosen = [int(np.argmax([c['tilt'] for c in cands]))]
    d = np.linalg.norm(feat - feat[chosen[0]], axis=1)
    while len(chosen) < n:
        k = int(np.argmax(d)); chosen.append(k); d = np.minimum(d, np.linalg.norm(feat - feat[k], axis=1))
    views = [cands[k] for k in chosen]
    views.sort(key=lambda c: (round(c['h'], 2), c['az'] if round(c['h'] * 50) % 2 == 0 else -c['az']))   # boustrophedon
    return np.array([c['q'] for c in views])


def fit_intrinsics(cam, obj_all, img_all, size, K0=None, free=False):
    """K (fx = fy, principal point free) and k1, k2 from the frames, with one round of outlier-frame rejection. free=True also
    frees fy: only identifiable with views rolled about the optical axis's other direction (wrist-roll views)."""
    f0 = K_NOMINAL[0, 0] if K0 is None else K0[0, 0]
    c0 = (size[0] / 2, size[1] / 2) if K0 is None else (K0[0, 2], K0[1, 2])
    K_init = np.array([[f0, 0, c0[0]], [0, f0, c0[1]], [0, 0, 1.0]])
    return cam.calibrate_sets(obj_all, img_all, size, K_init, extra_flags=0 if free else cv2.CALIB_FIX_ASPECT_RATIO)


def estimate_mount(q_all, obj_all, img_all, K, dist):
    """Camera mount T_EC from each view: T_EC = inv(T_BE(q)) inv(T_CM), mat frame = base frame, nominal kinematics.
    Returns (T_EC 4x4, rotation spread deg, translation spread m)."""
    Rs, ts = [], []
    for q, o, i in zip(q_all, obj_all, img_all):
        ok, rv, tv = cv2.solvePnP(o, i, K, dist)
        if not ok:
            continue
        T_CM = np.eye(4); T_CM[:3, :3] = cv2.Rodrigues(rv)[0]; T_CM[:3, 3] = tv.ravel()
        T = np.linalg.inv(fk_view(q)) @ np.linalg.inv(T_CM); Rs.append(T[:3, :3]); ts.append(T[:3, 3])
    R = Rot.from_matrix(Rs).mean().as_matrix(); t = np.median(ts, axis=0)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    spread = float(np.mean([Rot.from_matrix(R.T @ Ri).magnitude() for Ri in Rs]))
    return T, np.degrees(spread), float(np.mean(np.linalg.norm(np.array(ts) - t, axis=1)))


def describe_mount(T):
    z = T[:3, 2]
    return (f'optical axis {np.degrees(np.arctan2(-z[2], z[0])):.1f} deg below the gripper axis, '
            f'{np.degrees(np.arcsin(np.clip(z[1], -1, 1))):.1f} deg sideways; camera at {np.round(T[:3, 3] * 1e3, 1)} mm in {{E}}')


class Viewer:
    """Live camera window, kept on top and refreshed constantly (during moves, settling and capture). show=False: no-op."""

    def __init__(self, cam, show, title='ELEC3901 automatic camera calibration (q to stop)'):
        self.cam, self.on, self.win, self.stop, self.kept = cam, show, title, False, 0
        if show:
            cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.win, 960, 540)
            try:
                cv2.setWindowProperty(self.win, cv2.WND_PROP_TOPMOST, 1)      # in front of VS Code / Jupyter
            except cv2.error:
                pass

    def frame(self, frame=None, lines=(), flash=False):
        """Show frame (grabbed now if None) with markers outlined and status lines. Returns the frame."""
        if frame is None:
            frame = self.cam.read(flush=1)
        if not self.on:
            return frame
        vis = frame.copy(); c, i = self.cam.detect(frame)
        if len(i):
            cv2.aruco.drawDetectedMarkers(vis, c, i.reshape(-1, 1))
        if flash:
            cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 0), 14)
        for k, txt in enumerate([f'kept {self.kept}   {len(i)} markers in view'] + list(lines)):
            cv2.putText(vis, txt, (12, 36 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 5)
            cv2.putText(vis, txt, (12, 36 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
        cv2.imshow(self.win, vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.stop = True
        return frame

    def close(self):
        if self.on:
            cv2.destroyWindow(self.win); cv2.waitKey(1)


def move_and_watch(arm, q, viewer, label, tol=0.02, still_tol=0.003, still_s=0.3, timeout=8.0, roll=None):
    """Command q and keep the live view running until the joints are within tol (rad) of it, OR have stopped moving
    (< still_tol rad for still_s): with the factory gains (I = 0) a gravity-loaded joint stops 1-4 deg short and never gets
    within tol, which used to cost the full timeout at every view."""
    if roll is not None:
        arm.set_roll(roll)                                       # moves with the joints (same speed profile)
    arm.move_q(q, wait=False); t0 = time.time()
    q_prev, t_still = arm.read_q(), None
    while time.time() - t0 < timeout:
        if hasattr(arm, 'step'):                                 # twin: the physics only advances when stepped
            arm.step(20)
        viewer.frame(lines=[label, 'moving'])
        q_now = arm.read_q()
        if np.max(np.abs(q_now - np.asarray(q))) < tol or viewer.stop:
            break
        if np.max(np.abs(q_now - q_prev)) < still_tol:
            t_still = t_still or time.time()
            if time.time() - t_still > still_s and time.time() - t0 > 0.5:
                break
        else:
            t_still = None
        q_prev = q_now
        if not viewer.on and not hasattr(arm, 'step'):
            time.sleep(0.05)
    if hasattr(arm, 'step'):
        arm.move_q(q, wait=True)                                 # twin: finish the move and settle
    if roll is not None:
        arm.set_roll(roll, wait=True)


def wait_static(arm, viewer=None, label='', tol=0.002, dt=0.1, timeout=1.5):
    """Block until successive joint readings differ by < tol rad (servos settled); the live view keeps running."""
    t0 = time.time(); q_prev = arm.read_q()
    while time.time() - t0 < timeout:
        t1 = time.time()
        while time.time() - t1 < dt:
            if viewer is not None and viewer.on:
                viewer.frame(lines=[label, 'settling'])
            else:
                time.sleep(dt); break
        q = arm.read_q()
        if np.max(np.abs(q - q_prev)) < tol:
            return q
        q_prev = q
    return arm.read_q()


def grab_static(cam, corners_M, still_px=0.3, accept_px=1.0, tries=12, n_avg=3, viewer=None, label=''):
    """Frames until two consecutive ones agree (median corner movement < still_px: webcam corners jitter by a few tenths of
    a pixel even when nothing moves). If that never happens within `tries`, the stillest pair is accepted if its median
    movement is < accept_px. Returns (obj, img, ids, frame, jitter_px) with corners averaged over n_avg frames, or None."""
    viewer = viewer or Viewer(cam, False)
    def det(frame):
        c, ids = cam.detect(frame)
        return {int(i): cc.reshape(4, 2) for cc, i in zip(c, ids) if int(i) in corners_M}
    prev = det(viewer.frame(cam.read(flush=3), [label, 'waiting for a still image']))
    best = None
    for _ in range(tries):
        frame = cam.read(flush=1); now = det(frame)
        common = sorted(set(now) & set(prev))
        jit = float(np.median(np.concatenate([np.linalg.norm(now[i] - prev[i], axis=1) for i in common]))) if common else np.inf
        viewer.frame(frame, [label, f'waiting for a still image: jitter {min(jit, 99):.2f} px (need < {still_px})'])
        if common and (best is None or jit < best[0]):
            best = (jit, now, frame)
        if jit < still_px:
            break
        prev = now
    if best is None or best[0] >= accept_px:
        return None
    jit, now, frame = best
    stack = [now] + [det(viewer.frame(None, [label, 'averaging'])) for _ in range(n_avg - 1)]
    ids = [i for i in sorted(now) if all(i in s for s in stack)]
    if not ids:
        return None
    img = np.vstack([np.mean([s[i] for s in stack], axis=0) for i in ids]).astype(np.float32)
    obj = np.vstack([corners_M[i] for i in ids]).astype(np.float32)
    return obj, img, ids, frame, jit




def prepare_gripper_and_roll(arm, verbose=True, min_open=0.8):
    """At home: recover a tripped wrist roll / gripper (XL330 'input voltage' trips seen on the first arm), roll to zero,
    open the gripper and CHECK it opened and both hold torque. Raises before any view is taken if not: a closed gripper
    hides the mat, and a tripped roll lets the gripper-mounted camera swing, so every view would be invalid."""
    extra = [i for i in (15, 16)]
    if hasattr(arm, 'recover'):
        arm.recover(extra)
    if hasattr(arm, 'set_roll'):
        arm.set_roll(0.0, wait=True)
    arm.set_gripper('open', wait=True)
    if not hasattr(arm, 'hardware_errors') or getattr(arm, 'simulate', False):   # twin / simulated OMX: nothing to check
        return
    time.sleep(0.3)
    opening = arm.read_extra()[1]
    torque = {i: arm._read(i, 64, 1) for i in extra}
    errs = arm.hardware_errors(extra)
    if opening < min_open or errs or not all(torque.values()):
        raise RuntimeError(f'gripper not ready: opening {opening:.2f} (need >= {min_open}), torque {torque}, errors {errs or "none"}. '
                           'Nothing was measured. Check the XL330 supply (6.0 V bus, 7.0 V trip) and run again.')
    if verbose:
        print(f'gripper open ({opening:.2f}) and wrist roll holding; starting the views')


def roll_and_gripper_ok(arm):
    """False if the wrist roll or gripper has tripped since the preflight (their torque is cut on a voltage error)."""
    if not hasattr(arm, 'hardware_errors'):
        return True
    return not arm.hardware_errors([15, 16])


def capture(arm, cam, views, viewer, stage, min_markers, still_px, verbose):
    """Visit each view, keep static frames with >= min_markers. Returns lists (obj, img, ids, q, jitter) and the image size."""
    obj_all, img_all, id_all, q_all, jit_all, size = [], [], [], [], [], SIZE
    for k, v in enumerate(views):
        label = f'{stage}: view {k + 1}/{len(views)}'
        q, roll = np.asarray(v[:4], float), (float(v[4]) if len(v) > 4 else None)
        if roll is not None:
            label += f' roll {np.degrees(roll):+.0f} deg'
        move_and_watch(arm, q, viewer, label, roll=roll)
        q_meas = wait_static(arm, viewer, label)
        if not roll_and_gripper_ok(arm):
            raise RuntimeError(f'{label}: the wrist roll or gripper tripped ({arm.hardware_errors([15, 16])}) - the camera is no '
                               'longer held, so the run stops here; nothing is saved')
        if roll is not None:
            q_meas = np.append(q_meas, arm.read_extra()[0])         # measured roll becomes a 5th column
        if hasattr(cam, 'drain') and hasattr(cam, 'cap'):
            cam.drain(0.3)                                          # webcams queue frames: drop any from before the stop (lag 0.1-0.2 s)
        got = grab_static(cam, cam.corners_M, still_px=still_px, viewer=viewer, label=label)
        status = 'no still image (jitter too high) - skipped'
        if got is not None:
            obj, img, ids, frame, jit = got; size = frame.shape[1::-1]
            if len(ids) >= min_markers:
                obj_all.append(obj); img_all.append(img); id_all.append(np.array(ids)); q_all.append(q_meas)
                jit_all.append(jit); viewer.kept += 1
                status = f'kept: {len(ids)} markers, jitter {jit:.2f} px'
            else:
                status = f'only {len(ids)} markers - skipped'
            viewer.frame(frame, [label, status], flash=len(ids) >= min_markers)
        if verbose:
            print(f'  {label}: {status}' + ' ' * 10, end='\r')
        if viewer.stop:
            print('\n  stopped by the user (q)'); break
    if verbose:
        print()
    return obj_all, img_all, id_all, q_all, jit_all, size


def auto_calibrate(arm, cam, arm_id='OMX-00', views=None, K0=None, profile=(60, 20), still_px=0.3, n_views=18,
                   show=False, save_dir='.', verbose=True, gains='standard', open_gripper=True, rolls_deg=None,
                   stage1_only=False, stages=2, show_mount=True):
    """Bootstrap near home, measure the mount, visit the planned views, fit K and distortion (see the module docstring).
    open_gripper=True opens the gripper first (closed fingers hide the mat). rolls_deg (e.g. (0, 35, -35)): stage 1 cycles
    the wrist roll through these angles (the camera rolls with the gripper), which makes fy and cy identifiable, so the fit is
    also run fully free; without it the roll stays masked at zero. stages=1: stage 1 only, then the usual final fit and
    saves (the lab notebooks: on the first arm stage 1 alone gave f within 0.4 % of the two-stage fit, same mount, in about
    half the time, 30 Sep 2026); stages=2 (default, staff baselines) adds the planned views. stage1_only=True stops after
    stage 1 without the final fit or the session files (test runs). show_mount=False: do not print the measured mount (the lab
    notebooks: students measure it themselves in Lab 2; it is still saved in camera_autocal_<arm_id>.npz).
    Sets cam.K / cam.dist; saves camera_K_session_<arm_id>.npz and camera_autocal_<arm_id>.npz. Torque is switched ON and
    the arm moves: keep the workspace clear. show=True: a live camera window, on top, the whole time. Returns (K, dist, rms).
    views: skip the bootstrap and planning and use these joint configurations."""
    viewer = Viewer(cam, show)
    gains_before = getattr(arm, 'gains', 'standard')
    if hasattr(arm, 'set_gains'):
        arm.set_gains(gains)        # 'standard' by default: 'precision' made the first arm shake (30 Sep 2026)
    arm.torque(True); arm.set_profile(*profile)
    if open_gripper and hasattr(arm, 'set_gripper'):
        # closed fingers block the camera's view of the mat - but open them at HOME: at the folded rest pose the gripper is
        # beside the base and opening it drives the fingers into the base (first arm, 30 Sep 2026)
        arm.home(wait=True)                  # from rest this unfolds via REST_VIA_Q (a direct move could dip into the table)
        prepare_gripper_and_roll(arm, verbose)
    t0 = time.time()
    try:
        if views is None:
            b = capture(arm, cam, bootstrap_views(rolls_deg=rolls_deg), viewer, 'stage 1 (aimed)', 5, still_px, verbose)
            if viewer.stop:
                raise KeyboardInterrupt('stopped by the user (q) in stage 1: nothing fitted or saved')
            if len(b[0]) < 6:                                        # mount far from MOUNT_PRIOR_DEG: the mount-independent sweep
                if verbose:
                    print(f'stage 1: only {len(b[0])} aimed views saw >= 5 markers - the mount is far from the '
                          f'{MOUNT_PRIOR_DEG:.0f} deg guess; trying the pitch sweep')
                b2 = capture(arm, cam, bootstrap_views(rolls_deg=rolls_deg, mount_deg=None), viewer, 'stage 1 (sweep)', 5,
                             still_px, verbose)
                b = tuple(b[j] + b2[j] for j in range(5)) + (b2[5],)
                if viewer.stop:
                    raise KeyboardInterrupt('stopped by the user (q) in stage 1: nothing fitted or saved')
            if len(b[0]) < 6:
                raise RuntimeError(f'stage 1: only {len(b[0])} views saw >= 5 markers - check the camera, focus, lighting and mat')
            K1, dist1, rms1, _ = fit_intrinsics(cam, b[0], b[1], b[5], K0)
            T_ec, sp_deg, sp_m = estimate_mount(b[3], b[0], b[1], K1, dist1)
            if rolls_deg:
                Kf, df, rmsf, _ = fit_intrinsics(cam, b[0], b[1], b[5], K0, free=True)
                Tf, spf, spfm = estimate_mount(b[3], b[0], b[1], Kf, df)
                if verbose:
                    print(f'stage 1 FREE fit (fx, fy, c, k1, k2 all free; roll views): rms {rmsf:.2f} px, f {Kf[0, 0]:.0f}/{Kf[1, 1]:.0f} px, '
                          f'c ({Kf[0, 2]:.0f}, {Kf[1, 2]:.0f}), k1 {df.ravel()[0]:+.3f}\n  mount: {describe_mount(Tf)} '
                          f'(spread {spf:.1f} deg, {spfm * 1e3:.0f} mm)')
            if verbose:
                print(f'stage 1: {len(b[0])} views, rms {rms1:.2f} px, f {K1[0, 0]:.0f} px, c ({K1[0, 2]:.0f}, {K1[1, 2]:.0f}), '
                      f'k1 {dist1.ravel()[0]:+.2f}' + (f'\n  mount: {describe_mount(T_ec)} (spread {sp_deg:.1f} deg, {sp_m * 1e3:.0f} mm)'
                                                       if show_mount else ''))
            if stage1_only:
                obj_all, img_all, id_all, q_all, jit_all = b[:5]; size = b[5]
                move_and_watch(arm, HOME, viewer, 'returning home', roll=0.0 if rolls_deg else None)
                np.savez(os.path.join(save_dir, f'camera_autocal_stage1_{arm_id}.npz'), K=K1, dist=dist1, T_EC=T_ec,
                         q=np.array([np.pad(q, (0, 5 - len(q))) for q in q_all]),
                         obj=np.array(obj_all, dtype=object), img=np.array(img_all, dtype=object))
                return K1, dist1, rms1
            if stages == 1:
                obj_all, img_all, id_all, q_all, jit_all = b[:5]; size = b[5]
            else:
                views = calibration_views(n=n_views, K=K1, dist=dist1, T_ec=T_ec)
                p = capture(arm, cam, views, viewer, 'stage 2 (planned)', 8, still_px, verbose)
                if viewer.stop:
                    raise KeyboardInterrupt('stopped by the user (q) in stage 2: nothing fitted or saved')
                obj_all, img_all, id_all, q_all, jit_all = (b[j] + p[j] for j in range(5)); size = p[5]
        else:
            obj_all, img_all, id_all, q_all, jit_all, size = capture(arm, cam, views, viewer, 'views', 8, still_px, verbose)
        move_and_watch(arm, HOME, viewer, 'returning home', roll=0.0 if rolls_deg else None)
    except BaseException:
        try:                                                        # best effort: do not leave the arm out over the mat
            arm.move_q(HOME, wait=True, roll=0.0 if rolls_deg else None); print('\n  error: arm returned home (torque ON)')
        except Exception:
            print('\n  error: could NOT return the arm home - it is where it stopped, torque ON')
        raise
    finally:
        viewer.close()
        if hasattr(arm, 'set_gains'):
            try:
                arm.set_gains(gains_before)
            except Exception as e:
                print('could not restore the gains:', e)
    if len(obj_all) < 8:
        raise RuntimeError(f'only {len(obj_all)} usable views - check the camera, focus, lighting and the mat')
    K, dist, rms, kept = fit_intrinsics(cam, obj_all, img_all, size, K0)
    cam.K, cam.dist = K, dist
    errs = cam.frame_errors(obj_all, img_all, K, dist)
    good = [j for j, e in enumerate(errs) if e <= max(3.0, 3 * np.median(errs))]   # same rule as the fit's outlier rejection
    T_ec, sp_deg, sp_m = estimate_mount([q_all[j] for j in good], [obj_all[j] for j in good], [img_all[j] for j in good], K, dist)
    if verbose:
        print(f'calibrated from {len(obj_all)} static views in {time.time() - t0:.0f} s'
              f'{f" ({len(obj_all) - kept} rejected as outliers)" if kept < len(obj_all) else ""}: rms {rms:.2f} px, '
              f'f = {K[0, 0]:.0f}/{K[1, 1]:.0f} px, c = ({K[0, 2]:.0f}, {K[1, 2]:.0f}), k1 {dist.ravel()[0]:+.3f} k2 {dist.ravel()[1]:+.3f}')
        if show_mount:
            print(f'  camera mount: {describe_mount(T_ec)} (spread {sp_deg:.1f} deg, {sp_m * 1e3:.0f} mm, {len(good)} views)')
        print('  worst views:  view  error (px)  markers')
        for j in np.argsort(errs)[::-1][:5]:
            print(f'               {j + 1:4d}  {errs[j]:10.2f}  {len(id_all[j]):7d}')
    np.savez(os.path.join(save_dir, f'camera_K_session_{arm_id}.npz'), K=K, dist=dist, size=np.array(size), rms=rms)
    obj_arr, img_arr, id_arr = (np.empty(len(obj_all), dtype=object) for _ in range(3))
    for j in range(len(obj_all)):
        obj_arr[j], img_arr[j], id_arr[j] = obj_all[j], img_all[j], id_all[j]
    np.savez(os.path.join(save_dir, f'camera_autocal_{arm_id}.npz'), K=K, dist=dist, rms=rms, size=np.array(size), T_EC=T_ec,
             q=np.array(q_all), frame_errors=errs, obj=obj_arr, img=img_arr, ids=id_arr)
    return K, dist, rms


def main():
    ap = argparse.ArgumentParser(description='Automatic wrist-camera calibration (the arm moves: keep the workspace clear).')
    ap.add_argument('--port', default='auto', help="serial port of the OpenRB-150; 'auto' finds it by USB id")
    ap.add_argument('--arm-id', default='OMX-00', help='asset label on the base; tags the saved files')
    ap.add_argument('--camera', default='Innomaker', help="camera name (part of it) or number; see omx_camera.list_cameras()")
    ap.add_argument('--mat-scale', type=float, default=None, help='printed/nominal mat size (temporary; see omx_camera)')
    ap.add_argument('--views', type=int, default=18, help='planned views in stage 2')
    ap.add_argument('--show', action='store_true', help='live camera window the whole time')
    ap.add_argument('--sim', action='store_true', help='run on the MuJoCo digital twin and compare with its true K')
    ap.add_argument('--gains', default='standard', choices=['standard', 'precision'], help='servo gains during the run')
    ap.add_argument('--roll', default='', help="stage-1 wrist-roll angles in deg, e.g. '0,35,-35' (camera rolls with the gripper)")
    ap.add_argument('--stage1-only', action='store_true', help='stop after stage 1 (test run: no final fit, no session files)')
    ap.add_argument('--stages', type=int, default=2, choices=[1, 2], help='1: stage 1 only, as the lab notebooks; 2: both (baselines)')
    a = ap.parse_args()
    if a.sim:
        from omx_mujoco import OMXSim, SimCamera
        arm = OMXSim(view=False); cam = SimCamera(arm, use_true_K=False)
    else:
        from omx import OMX
        from omx_camera import Camera
        arm = OMX(a.port); cam = Camera(int(a.camera) if a.camera.isdigit() else a.camera, mat_scale=a.mat_scale)
        print('arm on', arm.port_name, '| camera', a.camera)
    try:
        rolls = [float(x) for x in a.roll.split(',')] if a.roll else None
        K, dist, rms = auto_calibrate(arm, cam, a.arm_id, n_views=a.views, show=a.show, gains=a.gains, rolls_deg=rolls,
                                      stage1_only=a.stage1_only, stages=a.stages)
        if a.sim:
            Kt = np.asarray(cam.true_K()[0])          # (K, dist) of the rendered camera
            print(f'twin: true f = {Kt[0, 0]:.1f} px, estimated {K[0, 0]:.1f}/{K[1, 1]:.1f} px '
                  f'({100 * (K[0, 0] / Kt[0, 0] - 1):+.2f} %), c error ({K[0, 2] - Kt[0, 2]:+.1f}, {K[1, 2] - Kt[1, 2]:+.1f}) px')
    finally:
        # torque stays ON: the arm holds its home pose. Switching it off would drop the arm under gravity - support it (or
        # move it to a low rest pose) before arm.torque(False).
        print('torque is ON - support the arm before switching torque off (it drops under gravity)')
        arm.close(); cam.close()


if __name__ == '__main__':
    main()
