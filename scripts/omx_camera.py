"""
omx_camera.py - wrist camera as a measuring instrument for ELEC3901.

Frames
  {M}  mat frame (origin at the nominal joint-1 axis, printed on the mat)
  {C}  camera frame, OpenCV convention: z along the optical axis, x right, y down in the image
  {E}  end-effector frame of omx_model (x along link 4)
  {B}  OMX base frame

The camera sees ArUco markers whose corners are known in {M} (mat_markers.json).
solvePnP over every visible corner gives T_CM (pose of {M} in {C}). Chaining
    T_BM = T_BE(q) * T_EC * T_CM
lets you (a) validate kinematics, (b) measure the tip position in {M} independently
of the kinematic chain: p_M(tip) = T_CM^-1 * T_EC^-1 * p_E(tip).

Intrinsics: students use a DIFFERENT ARM EACH WEEK and the manual focus ring may
have moved, so intrinsics are re-estimated at the start of every session using the
mat itself as the calibration target (Camera.calibrate_from_mat, ~30 s with torque
off). A per-arm baseline camera_K_<ARM_ID>.npz from the technicians, if present,
is used as the initial guess. calibrate_intrinsics.py (checkerboard) is the
technician tool for producing those baselines.

Usage
    from omx_camera import Camera
    cam = Camera(0)
    cam.calibrate_from_mat(save_as='camera_K_session.npz')   # every session
    T_CM, n = cam.mat_pose()          # SE3, number of markers used (0 => failed)
    cam.show()                        # annotated live view (press q)

NOT yet validated on the departmental cameras: check the device index and resolution.
"""
import json
import os
import numpy as np
import cv2
from spatialmath import SE3

HERE = os.path.dirname(os.path.abspath(__file__))
MAT_FILE = os.path.join(HERE, 'mat_markers.json')


def mat_scale_setting():
    """Printed-mat scale: measured / nominal length of a mat feature (e.g. the 50 mm grid: 48.35/50 = 0.967). Taken from the
    ELEC3901_MAT_SCALE environment variable, default 1.0. A stop-gap for testing on a mis-scaled print; reprint at 100% instead."""
    return float(os.environ.get('ELEC3901_MAT_SCALE', '1.0'))


def load_mat(path=MAT_FILE, scale=None):
    """Mat metadata and marker corners {id: (4,3) m}. scale: printed / nominal size (None: mat_scale_setting()). A printer that
    shrinks the page does so uniformly, so with the printed origin cross on the joint-1 axis every corner is scale x nominal."""
    m = json.load(open(path))
    scale = mat_scale_setting() if scale is None else float(scale)
    if abs(scale - 1.0) > 1e-9:
        print(f'omx_camera: mat scale {scale:.4f} (ELEC3901_MAT_SCALE) - marker positions and size scaled; reprint at 100% and remove it')
        m = dict(m, marker_side_m=m['marker_side_m'] * scale, scale=scale)
    corners = {int(k): scale * np.array(v['corners_m'], dtype=np.float64) for k, v in m['markers'].items()}
    return m, corners


def list_cameras():
    """Camera names in the order OpenCV numbers them (Windows: DirectShow order, needs `pip install pygrabber`;
    Linux: /sys/class/video4linux). Returns [(index, name)]."""
    import sys, glob
    if sys.platform == 'win32':
        try:
            from pygrabber.dshow_graph import FilterGraph
        except ImportError as e:
            raise ImportError('selecting a camera by name on Windows needs: pip install pygrabber') from e
        return [(i, n.strip()) for i, n in enumerate(FilterGraph().get_input_devices())]
    out = []
    for p in sorted(glob.glob('/sys/class/video4linux/video*/name')):
        out.append((int(p.split('video4linux/video')[1].split('/')[0]), open(p).read().strip()))
    return out


def find_camera(name):
    """Index of the first camera whose name contains `name` (case-insensitive), e.g. 'Innomaker'."""
    cams = list_cameras()
    for i, n in cams:
        if name.lower() in n.lower():
            return i
    raise IOError(f'no camera called {name!r}; found: {[n for _, n in cams] or "none"}')


class Camera:
    def __init__(self, index=0, intrinsics=None, width=1280, height=720, mat_file=MAT_FILE, mat_scale=None):
        """index: an OpenCV camera number, or a (part of a) device NAME such as 'Innomaker' - safer on a laptop, whose
        own webcam is usually number 0."""
        import sys
        self.name = index if isinstance(index, str) else None
        if isinstance(index, str):
            index = find_camera(index)
        if sys.platform == 'win32' and self.name is not None:
            self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)          # DirectShow: same numbering as list_cameras()
        else:
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))   # compressed: full frame rate at 1280x720
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)                     # ask for no queue (not every backend honours it)
        if not self.cap.isOpened():
            raise IOError(f'camera {self.name or index} not found')
        self.K = self.dist = None
        if intrinsics is not None:
            self.load_intrinsics(intrinsics)
        self.meta, self.corners_M = load_mat(mat_file, mat_scale)     # mat_scale None: ELEC3901_MAT_SCALE or 1.0
        self.adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.meta['dictionary']))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.adict, params)

    def load_intrinsics(self, path):
        d = np.load(path); self.K, self.dist = d['K'], d['dist']; return d

    def calibrate_from_mat(self, n_frames=25, seconds=30, K0=None, min_markers=6, save_as=None, verbose=True,
                           show=False, still_px=None):
        """Per-session intrinsic calibration using the fiducial mat as a planar target.
        Move the arm by hand (torque OFF) so the camera sees the mat from many heights AND TILTS: vary the
        camera height between about 90 and 180 mm and tilt the wrist by at least +/-30 deg (aim for 40).
        Nearly vertical views at one height cannot determine the focal length (they come out 10-20 % wrong). Frames with fewer than
        min_markers visible are skipped. Returns (K, dist, rms_px). Aim for rms < 0.6 px.
        show: live window (markers outlined, frames kept, time left, sharpness; press q to stop early).
        still_px: keep a frame only if the markers moved less than this many px since the previous frame, i.e. the camera is
        held still (hand-held motion blur and rolling shutter otherwise cost pixels). 1.5 is a good value; None = off.
        After the fit, prints the reprojection error of every kept frame, worst first."""
        import time as _t
        obj_all, img_all, info, size = [], [], [], None
        t0 = _t.time(); last = None; prev = {}; flash = 0; win = 'ELEC3901 camera calibration (q to stop)'
        while len(obj_all) < n_frames and _t.time() - t0 < seconds:
            frame = self.read(flush=1)
            size = frame.shape[1::-1]
            corners, ids = self.detect(frame)
            sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
            now = {int(i): c.reshape(4, 2) for c, i in zip(corners, ids) if i in self.corners_M}
            common = set(now) & set(prev)
            motion = np.mean([np.linalg.norm(now[i] - prev[i], axis=1).mean() for i in common]) if common else np.inf
            prev = now
            status = f'{len(now)} markers'
            if len(now) < min_markers:
                status += f' (need {min_markers})'
            elif still_px is not None and motion > still_px:
                status += f'  HOLD STILL (moving {min(motion, 99):.1f} px)'
            else:
                obj = np.vstack([self.corners_M[i].astype(np.float32) for i in now])
                img = np.vstack([now[i].astype(np.float32) for i in now])
                # require the view to have changed noticeably since the last kept frame
                if last is None or np.mean(np.linalg.norm(img.mean(0) - last, axis=-1)) > 25:
                    obj_all.append(obj); img_all.append(img); last = img.mean(0); flash = 4
                    info.append(dict(markers=len(now), sharp=sharp, motion=motion))
                    if verbose: print(f'  frame {len(obj_all)}/{n_frames}: {len(now)} markers, sharpness {sharp:.0f}', end='\r')
                else:
                    status += '  move to a NEW view'
            if show:
                vis = frame.copy()
                if len(ids):
                    cv2.aruco.drawDetectedMarkers(vis, corners, ids.reshape(-1, 1))
                if flash:
                    cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 0), 12); flash -= 1
                for k, (txt, col) in enumerate([
                        (f'kept {len(obj_all)}/{n_frames}   time left {max(0, seconds - (_t.time() - t0)):.0f} s', (0, 255, 0)),
                        (status, (0, 255, 255) if 'HOLD' in status or 'need' in status else (255, 255, 255)),
                        (f'sharpness {sharp:.0f}   vary height 90-180 mm, tilt +/-30-40 deg', (255, 255, 255))]):
                    cv2.putText(vis, txt, (12, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
                    cv2.putText(vis, txt, (12, 32 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
                cv2.imshow(win, vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                _t.sleep(0.05)
        if show:
            cv2.destroyWindow(win); cv2.waitKey(1)
        if len(obj_all) < 8:
            raise RuntimeError(f'only {len(obj_all)} usable frames - move the camera over the mat, 90-180 mm up, tilting it')
        K, dist, rms, kept = self.calibrate_sets(obj_all, img_all, size, K0)
        if kept < len(obj_all) and verbose:
            print(f'  ({len(obj_all) - kept} frame(s) rejected as outliers)')
        self.K, self.dist = K, dist
        if verbose: print(f'\ncalibrated from {len(obj_all)} frames: rms {rms:.2f} px, f = {K[0,0]:.0f}/{K[1,1]:.0f} px, c = ({K[0,2]:.0f}, {K[1,2]:.0f})')
        if verbose:
            errs = self.frame_errors(obj_all, img_all, K, dist)
            print('  worst frames:  frame  error (px)  markers  sharpness')
            for k in np.argsort(errs)[::-1][:8]:
                print(f'                {k + 1:5d}  {errs[k]:10.2f}  {info[k]["markers"]:7d}  {info[k]["sharp"]:9.0f}')
            print('  All frames high -> focus, or a mat that is not flat (or printed with different x and y scales);'
                  ' a few high -> blur in those views.')
            self.last_calibration = dict(frame_errors=errs, info=info)
        if save_as:
            np.savez(save_as, K=K, dist=dist, size=np.array(size), rms=rms)
        return K, dist, rms

    @staticmethod
    def frame_errors(obj_all, img_all, K, dist):
        """RMS reprojection error (px) of each frame under intrinsics (K, dist), each frame's pose fitted on its own."""
        out = []
        for o, i in zip(obj_all, img_all):
            ok, rv, tv = cv2.solvePnP(o, i, K, dist)
            p = cv2.projectPoints(o, rv, tv, K, dist)[0].reshape(-1, 2)
            out.append(float(np.sqrt(np.mean(np.sum((p - i.reshape(-1, 2)) ** 2, 1)))))
        return np.array(out)

    @staticmethod
    def calibrate_sets(obj_all, img_all, size, K0=None, reject_px=3.0, extra_flags=0):
        """cv2.calibrateCamera over per-frame point sets with one round of outlier-frame rejection.
        A single mis-identified marker in one frame can wreck the whole fit; frames whose reprojection error
        exceeds max(reject_px, 3 x median) are dropped and the calibration repeated. Returns (K, dist, rms, n_kept)."""
        flags = cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST | extra_flags
        if K0 is not None:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        def _cal(O, I):
            return cv2.calibrateCamera(O, I, size, None if K0 is None else K0.copy(), None, flags=flags)
        rms, K, dist, rv, tv = _cal(obj_all, img_all)
        per = np.array([np.sqrt(np.mean(np.sum((cv2.projectPoints(o, r, t, K, dist)[0].reshape(-1, 2) - i) ** 2, 1)))
                        for o, i, r, t in zip(obj_all, img_all, rv, tv)])
        keep = [k for k, e in enumerate(per) if e <= max(reject_px, 3 * np.median(per))]
        if len(keep) < len(per) and len(keep) >= 8:
            rms, K, dist, rv, tv = _cal([obj_all[k] for k in keep], [img_all[k] for k in keep])
        return K, dist, rms, len(keep)

    def drain(self, seconds=0.5):
        """Throw frames away for `seconds`. Use after the arm stops: webcams queue frames, so a read straight after a move can
        return an image from BEFORE it (a fixed number of flushed frames is not enough at a low frame rate)."""
        import time as _t
        t0 = _t.time()
        while _t.time() - t0 < seconds:
            self.cap.grab()

    def read(self, flush=3):
        for _ in range(flush):                          # drop buffered frames so the image matches 'now'
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise IOError('frame grab failed')
        return frame

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        return corners, (ids.flatten() if ids is not None else np.array([], int))

    def mat_pose(self, frame=None, min_markers=2, return_reproj=False):
        """Pose of the mat in the camera frame, T_CM (SE3), from all visible markers.
        Returns (T_CM, n_markers) or (None, n) if fewer than min_markers are seen."""
        frame = self.read() if frame is None else frame
        corners, ids = self.detect(frame)
        obj, img = [], []
        for c, i in zip(corners, ids):
            if i in self.corners_M:
                obj.append(self.corners_M[i]); img.append(c.reshape(4, 2))
        n = len(obj)
        if n < min_markers:
            return (None, n, None) if return_reproj else (None, n)
        obj = np.vstack(obj); img = np.vstack(img).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, flags=cv2.SOLVEPNP_IPPE)
        ok, rvec, tvec = cv2.solvePnP(obj, img, self.K, self.dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)      # refine
        R, _ = cv2.Rodrigues(rvec)
        T_CM = SE3.Rt(R, tvec.flatten())
        if return_reproj:
            proj, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
            rms = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1)))
            return T_CM, n, rms
        return T_CM, n

    def mat_pose_avg(self, n_frames=5, **kw):
        """Average translation and orientation over several frames (reduces jitter)."""
        Ts = []
        for _ in range(n_frames):
            T, n = self.mat_pose(**kw)
            if T is not None:
                Ts.append(T)
        if not Ts:
            return None, 0
        t = np.mean([T.t for T in Ts], axis=0)
        # average rotation via the chordal mean (SVD of summed matrices)
        U, _, Vt = np.linalg.svd(sum(T.R for T in Ts))
        R = U @ Vt
        if np.linalg.det(R) < 0:
            U[:, -1] *= -1; R = U @ Vt
        return SE3.Rt(R, t), len(Ts)

    def show(self, T_EC=None):
        """Live annotated view. Press q to quit."""
        while True:
            frame = self.read(flush=1)
            corners, ids = self.detect(frame)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1) if len(ids) else None)
            T, n, rms = self.mat_pose(frame, return_reproj=True)
            if T is not None:
                rvec, _ = cv2.Rodrigues(T.R)
                cv2.drawFrameAxes(frame, self.K, self.dist, rvec, T.t, 0.05)
                cv2.putText(frame, f'{n} markers  reproj {rms:.2f} px  t_CM = {np.round(T.t*1e3,1)} mm',
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow('ELEC3901 wrist camera', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def focus_check(self, frame=None):
        """Sharpness (variance of Laplacian) and marker count. Compare against the value
        you recorded when the focus ring was taped; a large drop means the ring has moved
        or the working distance is wrong."""
        frame = self.read() if frame is None else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
        _, ids = self.detect(frame)
        return sharp, len(ids)

    def close(self):
        self.cap.release()


# ------------------------------------------------------------------------------
def tip_in_mat(T_CM, T_EC, p_tip_E):
    """Tip position in the mat frame from a camera measurement (independent of q).
    T_CM: SE3 from Camera.mat_pose; T_EC: camera in end-effector frame (CAD, refined in Lab 9);
    p_tip_E: tip offset in the end-effector frame (e.g. [0,0,0] for the model's EE point)."""
    return (T_CM.inv() * T_EC.inv()) * np.asarray(p_tip_E, float)


def predicted_T_CM(robot, q, T_EC, T_BM):
    """Kinematic prediction of what the camera should measure."""
    return (robot.fkine(q) * T_EC).inv() * T_BM


def pose_error(T_meas, T_pred):
    """(translation error in m, rotation error in rad) between two SE3 poses."""
    dT = T_pred.inv() * T_meas
    return np.linalg.norm(dT.t), np.linalg.norm(dT.rpy())
