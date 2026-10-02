"""
omx_mujoco.py - MuJoCo digital twin of the ELEC3901 work cell (OMX + base plate + mat + wrist camera).

    pip install mujoco          (3.x; CPU only, Windows/macOS/Linux)

Drop-in replacement for the hardware wrapper:

    from omx_mujoco import OMXSim, SimCamera
    arm = OMXSim(view=True)             # opens the MuJoCo viewer; view=False for headless
    cam = SimCamera(arm)                # same API as omx_camera.Camera, frames rendered from the wrist camera
    arm.torque(True); arm.set_profile(60, 20); arm.move_q([0, 0.5, -0.2, 1.27], wait=True)
    T_CM, n = cam.mat_pose()            # ArUco pipeline on the rendered image
    T_CM_true = arm.true_T_CM()         # ground truth for checking calibration
    arm.record('run.mp4')               # MP4 of the wrist camera and a scene camera side by side (same size)
    ...                                 # every move_q/stream/settle adds frames at 25 fps of simulated time
    arm.stop_recording()                # (also called by arm.close())

What is modelled
  * Arm: the ROBOTIS OMX-F (OpenMANIPULATOR-X follower). Meshes and inertials come from the ROBOTIS MuJoCo menagerie
    (`robotis_omx/omx.xml`, Apache-2.0, copied to labs/assets/omx_f); the body chain is re-expressed with the
    nominal ETS parameters of omx_model.py so that kin_error works. Joint 5 (wrist roll) and the gripper fingers are
    fixed bodies: the course treats the arm as 4-DOF; the fingers are fixed open (FINGER_OPEN_DEG), as for all camera work.
  * Actuation: a torque motor per joint driven by a Python position loop at the physics rate (default 1 kHz),
    with profile-velocity rate limiting, torque saturation at the servo stall torque (XL430 1.4 N m on joints 1-3,
    XL330 0.52 N m on joint 4), joint damping / Coulomb friction / rotor armature. This mirrors a Dynamixel in
    position mode; kp, kv, friction and armature are the parameters Lab 8 identifies.
  * Work cell: bench, base plate A, top frame B, the printed mat as a textured plane at z = 0 (base frame),
    optional mat placement error.
  * Wrist camera: MuJoCo camera at T_EC_TWIN (the first arm's measured mount, 62 deg below the gripper axis; the model's
    T_EC is the CAD value, T_EC_true=T_EC_NOMINAL gives a camera that matches it) with fovy chosen to match a given focal
    length; OpenCV-convention intrinsics available as SimCamera.K. Zero distortion.
  * Hidden truth: pass kin_error / joint_offsets / T_EC_error / mat_error to make the "real" arm differ from the
    nominal model, then test the calibration pipeline against ground_truth().

Not modelled: gripper kinematics/contacts (gripper() only records the state), contact with anything placed in the cell (static
visual geometry only), servo bus latency, backlash.

Coordinate conventions
  * World frame = OMX base frame {B} of omx_model.py: origin at the joint-1 axis on the plate top surface (mat
    surface), x forward, z up. Body frames coincide with the ETS frames at q = 0 (all parallel to {B}).
  * MuJoCo cameras look along -z with +y up in the image; OpenCV cameras look along +z with +y down. The two
    differ by a 180 deg rotation about x, handled here.

First run 21 Sep 2026 on MuJoCo 3.13 (Windows): compiles, servos, renders. sim_smoke_test.py passed all its steps on 21 Sep 2026:
mat_flip=(False, False) confirmed (0.6 px overhead projection error), wrist-camera mat pose 0.69 mm / 0.035 deg, two-stage
calibration on a twin with hidden errors: tip error 4.95 mm nominal -> 0.35 mm after the touch test.
"""
import os
import time
import numpy as np
from scipy.spatial.transform import Rotation as Rot

try:
    import mujoco
    _MJ = True
except ImportError:
    _MJ = False

HERE = os.path.dirname(os.path.abspath(__file__))
MAT_PNG = os.path.join(HERE, 'ELEC3901_mat_A3.png')
MESH_DIR = os.path.join(HERE, 'assets', 'omx_f')
MESHES = ['follower_01_base', 'follower_02_base_tilt_Revised', 'follower_03_middle_verticle', 'follower_04_middle_horizontal',
          'follower_05_tip', 'follower_06_pan_Revised', 'follower_07_gripper_motorized', 'follower_08_gripper_gear']

# nominal geometry (must match omx_model.py) - OMX-F, from the menagerie body chain
NOMINAL = dict(d1=0.0975, l2z=0.11315, l2x=0.0415, l3=0.162, l4=0.123)
Q_LIM = np.array([[-2.618, 2.618], [-1.868, 1.658], [-1.571, 1.676], [-1.745, 1.745]])
HOME_Q = np.radians([0.0, -60.0, 60.0, 30.0])                  # = omx_model.HOME_Q (home pose)
REST_Q = np.radians([-30.0, -55.0, 95.0, -5.0])                # = omx_model.REST_Q (rest pose)
REST_VIA_Q = np.radians([0.0, -60.0, 60.0, -5.0])              # = omx_model.REST_VIA_Q
# fixed offsets inside the OMX-F chain (m): J1 flange height above the mounting plane, base-mesh origin behind the J1 axis,
# roll axis (locked joint 5) and finger root along link 4
J1_Z = 0.034; BASE_X = -0.01125; ROLL_X = 0.0287; FINGER_X = 0.0295; FINGER_Y = (0.0075, -0.0108)
FINGER_OPEN_DEG = 30.0  # fingers fixed OPEN, turned about their pivots (URDF gripper_joint_1, _2 mirrored): as on the real arm, the
                        # gripper stays open for camera work; closed (0), they fill the view of a camera at the measured mount
STALL_TORQUE = np.array([1.4, 1.4, 1.4, 0.52])       # XL430-W250 x3, XL330-M288 (N m)
STALL_CURRENT_A = np.array([1.3, 1.3, 1.3, 1.47])
R_EC = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])          # camera axes in {E} (columns), see omx_model.py
T_EC_NOMINAL = np.eye(4); T_EC_NOMINAL[:3, :3] = R_EC; T_EC_NOMINAL[:3, 3] = [-0.030, 0.0, 0.040]   # = omx_model.T_EC (CAD)
# The twin's TRUE camera mount, as measured on the first arm (baseline calibration, 30 Sep 2026: optical axis 61.7-61.9 deg
# below the gripper axis, camera at ~(-23, -2, 20) mm in {E}): the students' model keeps the CAD T_EC and, as on the real
# arm, has to find this (Lab 2 Part D, refined in Lab 9). Pass T_EC_true=T_EC_NOMINAL for a twin whose camera matches the model.
# Copied in tasks.T_EC_TWIN_A (synthetic camera frames); check_consistency keeps them equal.
TWIN_MOUNT_DEG = 62.0
_a = np.radians(TWIN_MOUNT_DEG)
T_EC_TWIN = np.eye(4)
T_EC_TWIN[:3, :3] = [[0, -np.sin(_a), np.cos(_a)], [-1, 0, 0], [0, -np.cos(_a), -np.sin(_a)]]
T_EC_TWIN[:3, 3] = [-0.023, -0.002, 0.020]
TORQUE_PER_AMP = STALL_TORQUE / STALL_CURRENT_A     # per joint
RAD_PER_TICK = 2 * np.pi / 4096
VEL_UNIT = 0.229 * 2 * np.pi / 60          # rad/s per Dynamixel profile-velocity unit

# work cell (m) - must match make_baseplate.py / make_mat.py
SHEET = (0.420, 0.297); MAT_ORIGIN_ON_SHEET = (0.060, 0.1485)
PLATE = (0.480, 0.340, 0.006); FRAME_T = 0.003; FEET_H = 0.010; BENCH_T = 0.040
MAT_CENTRE = (SHEET[0] / 2 - MAT_ORIGIN_ON_SHEET[0], SHEET[1] / 2 - MAT_ORIGIN_ON_SHEET[1])   # (0.15, 0.0) in {B}


def _quat_wxyz(R):
    q = Rot.from_matrix(R).as_quat()          # x y z w
    return f"{q[3]:.6f} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}"


def _T(R=None, t=None):
    T = np.eye(4)
    if R is not None: T[:3, :3] = R
    if t is not None: T[:3, 3] = t
    return T


def _small_T(rotvec, t):
    return _T(Rot.from_rotvec(rotvec).as_matrix(), t)


def _object_geoms(objects):
    """MJCF for static objects placed in the cell: any list with a .mjcf() method (and .truth() for ground_truth()). '' if none."""
    return objects.mjcf() if objects else ''


def build_mjcf(kin=NOMINAL, T_EC=T_EC_NOMINAL, mat_T=np.eye(4), cam_res=(1280, 720), cam_f_px=950.0,
               damping=0.05, frictionloss=0.1, armature=0.01, tau_max=STALL_TORQUE, mat_flip=(False, False),
               timestep=0.001, objects=(), finger_open_deg=None):
    """Return the MJCF XML string for the cell. All lengths in metres. tau_max: per-joint motor limits (N m)."""
    d1, l2z, l2x, l3, l4 = (kin[k] for k in ('d1', 'l2z', 'l2x', 'l3', 'l4'))
    tau_max = np.broadcast_to(np.asarray(tau_max, float), (4,))
    meshes = "\n    ".join(f'<mesh name="{m}" file="{m}.stl" scale="0.001 0.001 0.001"/>' for m in MESHES)
    W, H = cam_res
    fovy = np.degrees(2 * np.arctan(H / (2 * cam_f_px)))
    # wrist camera in the link-4 body frame: EE point at (l4,0,0), then T_EC; MuJoCo camera = OpenCV camera * Rx(pi)
    _fo = np.radians(FINGER_OPEN_DEG if finger_open_deg is None else finger_open_deg)   # each finger turned about its pivot (z)
    R_mj = T_EC[:3, :3] @ np.diag([1, -1, -1])
    cam_pos = np.array([l4, 0, 0]) + T_EC[:3, 3]
    mat_pos = mat_T[:3, :3] @ np.array([MAT_CENTRE[0], MAT_CENTRE[1], 0]) + mat_T[:3, 3]
    px, py = PLATE[0] / 2 - 0.03 - MAT_ORIGIN_ON_SHEET[0] + 0.0, 0.0   # plate centre: mat window offset 30 mm inside plate
    px = MAT_CENTRE[0]                                                    # window centred on the plate
    z_plate = -PLATE[2] / 2                                               # plate top at z = 0
    z_bench = -PLATE[2] - FEET_H - BENCH_T / 2
    hf, vf = ('true' if mat_flip[0] else 'false'), ('true' if mat_flip[1] else 'false')
    lim = lambda i: f"{Q_LIM[i,0]:.4f} {Q_LIM[i,1]:.4f}"
    # frame B = four boxes around the A3 window (window inner edge at mat sheet edges)
    fx0, fx1 = MAT_CENTRE[0] - SHEET[0] / 2, MAT_CENTRE[0] + SHEET[0] / 2
    fy0, fy1 = -SHEET[1] / 2, SHEET[1] / 2
    Px0, Px1 = MAT_CENTRE[0] - PLATE[0] / 2, MAT_CENTRE[0] + PLATE[0] / 2
    Py0, Py1 = -PLATE[1] / 2, PLATE[1] / 2
    zf = FRAME_T / 2
    frame = f"""
    <geom name="frameL" type="box" size="{(fx0-Px0)/2:.4f} {PLATE[1]/2:.4f} {zf:.4f}" pos="{(Px0+fx0)/2:.4f} 0 {zf:.4f}" material="frame"/>
    <geom name="frameR" type="box" size="{(Px1-fx1)/2:.4f} {PLATE[1]/2:.4f} {zf:.4f}" pos="{(Px1+fx1)/2:.4f} 0 {zf:.4f}" material="frame"/>
    <geom name="frameF" type="box" size="{SHEET[0]/2:.4f} {(fy0-Py0)/2:.4f} {zf:.4f}" pos="{MAT_CENTRE[0]:.4f} {(Py0+fy0)/2:.4f} {zf:.4f}" material="frame"/>
    <geom name="frameB" type="box" size="{SHEET[0]/2:.4f} {(Py1-fy1)/2:.4f} {zf:.4f}" pos="{MAT_CENTRE[0]:.4f} {(Py1+fy1)/2:.4f} {zf:.4f}" material="frame"/>"""
    return f"""
<mujoco model="elec3901_omx_cell">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual>
    <global offwidth="{W}" offheight="{H}"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="mat_tex" type="2d" file="mat.png" hflip="{hf}" vflip="{vf}"/>
    <material name="mat" texture="mat_tex" texrepeat="1 1" texuniform="false" specular="0.05" shininess="0.1"/>
    <material name="plate" rgba="0.55 0.45 0.35 1"/>
    <material name="frame" rgba="0.62 0.52 0.40 1"/>
    <material name="bench" rgba="0.75 0.75 0.72 1"/>
    <material name="servo" rgba="0.15 0.15 0.15 1"/>
    <material name="link"  rgba="0.85 0.85 0.85 1"/>
    <material name="arm"   rgba="0.12 0.12 0.13 1" specular="0.3" shininess="0.4"/>
    {meshes}
  </asset>
  <default>
    <joint type="hinge" damping="{damping}" frictionloss="{frictionloss}" armature="{armature}"/>
    <geom contype="0" conaffinity="0"/>
    <default class="arm"><geom type="mesh" material="arm" density="0"/></default>
  </default>
  <worldbody>
    <light pos="0.3 0.2 1.2" dir="-0.2 -0.1 -1" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
    <light pos="-0.2 -0.4 1.0" dir="0.2 0.4 -1" diffuse="0.4 0.4 0.4" directional="false"/>
    <geom name="bench" type="box" size="0.6 0.45 {BENCH_T/2:.4f}" pos="{MAT_CENTRE[0]:.4f} 0 {z_bench:.4f}" material="bench"/>
    <geom name="plateA" type="box" size="{PLATE[0]/2:.4f} {PLATE[1]/2:.4f} {PLATE[2]/2:.4f}" pos="{MAT_CENTRE[0]:.4f} 0 {z_plate:.4f}" material="plate"/>
    {frame}
    <geom name="mat" type="plane" size="{SHEET[0]/2:.4f} {SHEET[1]/2:.4f} 0.01" pos="{mat_pos[0]:.5f} {mat_pos[1]:.5f} {mat_pos[2]+0.0003:.5f}" quat="{_quat_wxyz(mat_T[:3,:3])}" material="mat"/>
    {_object_geoms(objects)}
    <camera name="overhead" pos="{MAT_CENTRE[0]:.4f} 0 0.75" fovy="40"/>
    <!-- OMX-F (ROBOTIS menagerie meshes + inertials). World origin = joint-1 axis on the mounting plane. -->
    <body name="link0" pos="{BASE_X:.5f} 0 0">
      <inertial pos="-0.014412508 -0.0031498879 0.019276599" mass="0.22389051" diaginertia="0.00035680461 0.00021941932 0.00045575899"/>
      <geom mesh="follower_01_base" class="arm"/>
      <body name="link1" pos="{-BASE_X:.5f} 0 {J1_Z:.5f}">
        <inertial pos="-6.2236205e-06 0.0006049232 0.047418803" mass="0.065987041" diaginertia="2.1731667e-05 2.1351513e-05 1.1657955e-05"/>
        <joint name="j1" axis="0 0 1" range="{lim(0)}"/>
        <geom mesh="follower_02_base_tilt_Revised" class="arm"/>
        <body name="link2" pos="0 0 {d1 - J1_Z:.5f}">
          <inertial pos="0.018089916 0.00029778758 0.099005452" mass="0.087221471" diaginertia="0.00010307915 0.00012086327 4.1993677e-05"/>
          <joint name="j2" axis="0 1 0" range="{lim(1)}"/>
          <geom mesh="follower_03_middle_verticle" class="arm"/>
          <body name="link3" pos="{l2x:.5f} 0 {l2z:.5f}">
            <inertial pos="0.087398602 0.00041310065 -0.003582921" mass="0.083759129" diaginertia="1.5203101e-05 0.00022606158 0.00023066887"/>
            <joint name="j3" axis="0 1 0" range="{lim(2)}"/>
            <geom mesh="follower_04_middle_horizontal" class="arm"/>
            <body name="link4" pos="{l3:.5f} 0 0">
              <inertial pos="0.02323498 6.7735378e-05 0.0059761312" mass="0.029975063" diaginertia="5.099513e-06 7.0990471e-06 6.3119495e-06"/>
              <joint name="j4" axis="0 1 0" range="{lim(3)}"/>
              <geom mesh="follower_05_tip" class="arm"/>
              <body name="link5" pos="{ROLL_X:.5f} 0 0">
                <inertial pos="0.028072407 2.4135941e-05 0.013528914" mass="0.043810887" diaginertia="2.3694459e-05 2.3236924e-05 7.1806291e-06"/>
                <geom mesh="follower_06_pan_Revised" class="arm"/>
                <body name="finger1" pos="{FINGER_X:.5f} {FINGER_Y[0]:.5f} 0" euler="0 0 {_fo:.5f}">
                  <inertial pos="0.016164565 0.0012731032 0.00027291598" mass="0.012090677" diaginertia="2.7083287e-06 6.8733728e-06 4.6941881e-06"/>
                  <geom mesh="follower_07_gripper_motorized" class="arm"/>
                </body>
                <body name="finger2" pos="{FINGER_X:.5f} {FINGER_Y[1]:.5f} 0" euler="0 0 {-_fo:.5f}">
                  <inertial pos="0.016164565 0.0012731032 0.00027291598" mass="0.012090677" diaginertia="2.7083287e-06 6.8733728e-06 4.6941881e-06"/>
                  <geom mesh="follower_08_gripper_gear" class="arm"/>
                </body>
              </body>
              <site name="ee" pos="{l4:.5f} 0 0" size="0.003"/>
              <camera name="wrist" pos="{cam_pos[0]:.5f} {cam_pos[1]:.5f} {cam_pos[2]:.5f}" quat="{_quat_wxyz(R_mj)}" fovy="{fovy:.4f}"/>
              <geom type="box" size="0.012 0.012 0.008" pos="{cam_pos[0]:.5f} {cam_pos[1]:.5f} {cam_pos[2]:.5f}" mass="0.03" rgba="0.2 0.2 0.25 1"/>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="m1" joint="j1" ctrlrange="-{tau_max[0]:.3f} {tau_max[0]:.3f}"/>
    <motor name="m2" joint="j2" ctrlrange="-{tau_max[1]:.3f} {tau_max[1]:.3f}"/>
    <motor name="m3" joint="j3" ctrlrange="-{tau_max[2]:.3f} {tau_max[2]:.3f}"/>
    <motor name="m4" joint="j4" ctrlrange="-{tau_max[3]:.3f} {tau_max[3]:.3f}"/>
  </actuator>
</mujoco>
"""


class OMXSim:
    """Digital twin with the omx.OMX interface. Joint angles are reported like a servo would: q_reported =
    q_true - joint_offsets. Physics runs when you call move_q(wait=True), stream(), step() or settle()."""

    def __init__(self, view=False, kin_error=None, joint_offsets=None, T_EC_error=None, mat_error=None, T_EC_true=None,
                 kp=(40, 60, 40, 15), kv=(1.0, 1.5, 1.0, 0.4), tau_max=STALL_TORQUE, cam_res=(1280, 720),
                 cam_f_px=950.0, mat_flip=(False, False), objects=None, **mjcf_kw):
        if not _MJ:
            raise ImportError('pip install mujoco')
        self.kin = dict(NOMINAL); self.kin.update(kin_error or {})
        self.off = np.zeros(4) if joint_offsets is None else np.asarray(joint_offsets, float)
        self.T_EC_true = (T_EC_TWIN if T_EC_true is None else np.asarray(T_EC_true, float)) @ (
            _small_T(*T_EC_error) if T_EC_error else np.eye(4))
        self.mat_T = _small_T(*mat_error) if mat_error else np.eye(4)
        self.cam_res = cam_res
        self.objects = objects if objects is not None else []         # static objects in the cell (a list with .mjcf(), .truth())
        xml = build_mjcf(self.kin, self.T_EC_true, self.mat_T, cam_res, cam_f_px, tau_max=tau_max,
                         mat_flip=mat_flip, objects=self.objects, **mjcf_kw)
        assets = {'mat.png': open(MAT_PNG, 'rb').read()}
        for m in MESHES:
            assets[m + '.stl'] = open(os.path.join(MESH_DIR, m + '.stl'), 'rb').read()
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep
        self.kp, self.kv = np.array(kp, float), np.array(kv, float)
        self.tau_max = np.broadcast_to(np.asarray(tau_max, float), (4,)).copy()
        self.tau_ff = np.zeros(4)
        self.tau_limit = self.tau_max.copy()          # per-joint cap (current-based position mode analogue)
        self.q_goal = np.zeros(4); self.q_ref = np.zeros(4)
        self.vmax = 60 * VEL_UNIT                     # profile velocity (rad/s)
        self.torque_on = False
        self._grip_open = True
        self.viewer = None
        self._renderer = None
        self._rec = None                              # video recorder state, see record()
        self._t_sync = None                           # wall clock of the last step: reads catch the physics up (real time)
        mujoco.mj_forward(self.model, self.data)
        if view:
            from mujoco import viewer as mj_viewer        # not "import mujoco.viewer": that rebinds mujoco as a local
            self.viewer = mj_viewer.launch_passive(self.model, self.data)

    # ---- internals ----------------------------------------------------------
    def _servo_loop(self):
        q, qd = self.data.qpos[:4], self.data.qvel[:4]
        if self.torque_on:
            step = self.vmax * self.dt
            self.q_ref += np.clip(self.q_goal - self.q_ref, -step, step)
            tau = self.kp * (self.q_ref - q) - self.kv * qd + self.tau_ff
            tau = np.clip(tau, -self.tau_limit, self.tau_limit)
        else:
            tau = np.zeros(4)
        self.data.ctrl[:4] = tau

    def step(self, n=1):
        for _ in range(n):
            self._servo_loop()
            mujoco.mj_step(self.model, self.data)
            if self._rec is not None:
                self._rec['k'] += 1
                if self._rec['k'] >= self._rec['every']:
                    self._rec['k'] = 0; self.snapshot()
        if self.viewer is not None:
            self.viewer.sync()
        self._t_sync = time.time()

    def _catch_up(self, max_s=1.0):
        """Advance the physics to the wall clock (at most max_s at once), so non-blocking moves progress between reads the
        way the real arm does - e.g. step_log() or a streaming loop that only calls move_q() and read_q()."""
        if self._t_sync is None:
            return
        n = min(int((time.time() - self._t_sync) / self.dt), int(max_s / self.dt))
        if n > 0:
            self.step(n)

    def settle(self, seconds=1.0):
        self.step(int(seconds / self.dt))

    # ---- OMX-compatible API -------------------------------------------------
    def torque(self, on, ids=None):
        self._catch_up()
        if self._t_sync is None:
            self._t_sync = time.time()                   # run in real time from now: with torque off the arm drops, as on the bench
        self.torque_on = bool(on)
        if on:
            self.q_ref = self.data.qpos[:4].copy(); self.q_goal = self.q_ref.copy()

    def set_mode(self, mode, ids=None):
        pass

    def set_profile(self, velocity=60, acceleration=20, ids=None):
        self.vmax = (velocity if velocity > 0 else 1e9) * VEL_UNIT

    def set_gains(self, name='standard', ids=None):
        """Same call as omx.OMX; the twin's servo model is fitted separately (Lab 8), so the gain set is only recorded."""
        self.gains = name

    def set_pid(self, P=800, I=0, D=0, ids=None):
        """Dynamixel gain units are not physical; map the factory P=800 to the default kp and scale."""
        self.kp = np.array([40, 60, 40, 15], float) * (P / 800.0)
        self.kv = np.array([1.0, 1.5, 1.0, 0.4], float) * (1 + D / 2000.0)

    def set_goal_current(self, amps, ids=None):
        """Torque cap on selected joints (current-based position mode analogue). On the real OMX-F only joint 4 (XL330)
        supports this; the twin applies it to whichever joints you name so the idea can still be explored."""
        idx = range(4) if ids is None else [JOINT_INDEX[i] for i in ids]
        for j in idx:
            self.tau_limit[j] = min(abs(amps) * TORQUE_PER_AMP[j], self.tau_max[j])

    def set_feedforward(self, tau):
        self.tau_ff = np.asarray(tau, float)

    def read_q(self):
        self._catch_up()
        return self.data.qpos[:4] - self.off

    def read_qd(self):
        self._catch_up()
        return self.data.qvel[:4].copy()

    def read_current(self):
        """Applied motor torque as a current, like a Dynamixel present-current register (per-joint stall figures)."""
        self._catch_up()
        return self.data.ctrl[:4] / TORQUE_PER_AMP

    def read_load(self):
        """Applied torque as a fraction of stall torque (what the XL430 Present Load register reports)."""
        self._catch_up()
        return self.data.ctrl[:4] / STALL_TORQUE

    def read_torque(self):
        self._catch_up()
        return self.data.ctrl[:4].copy()

    def move_q(self, q, wait=False, tol=0.01, timeout=5.0, roll=None, gripper=None):
        q = np.clip(np.asarray(q, float), Q_LIM[:, 0], Q_LIM[:, 1])
        if roll is not None:
            self.set_roll(roll)
        if gripper is not None:
            self.set_gripper(gripper)
        self._catch_up()                                         # finish the time that passed before this command
        self.q_goal = q + self.off
        if self._t_sync is None:
            self._t_sync = time.time()
        if wait:
            n = 0
            while n * self.dt < timeout:
                self.step(10); n += 10
                if np.max(np.abs(self.data.qpos[:4] - self.q_goal)) < tol and np.max(np.abs(self.data.qvel[:4])) < 0.05:
                    break

    def stream(self, qs, dt):
        log_t, log_q = [], []
        n = max(1, int(round(dt / self.dt)))
        for k, q in enumerate(qs):
            self.move_q(q)
            log_t.append(k * dt); log_q.append(self.read_q())
            self.step(n)
        return np.array(log_t), np.array(log_q)

    def gripper(self, open_=True):
        self._grip_open = open_

    def set_gripper(self, g, wait=False, timeout=3.0):
        """Same call as omx.OMX: 'open', 'closed' or a fraction. The twin's fingers are not actuated; the state is recorded."""
        self._grip_open = (g == 'open') if isinstance(g, str) else float(g) > 0.5

    def set_roll(self, angle, wait=False, timeout=3.0):
        """Same call as omx.OMX. The twin keeps the wrist roll locked at zero (as the labs do); a non-zero request is refused."""
        if abs(float(angle)) > 1e-9:
            raise NotImplementedError('the twin does not model the wrist roll (joint 5): it stays at zero')

    def read_extra(self):
        return 0.0, 1.0 if getattr(self, '_grip_open', True) else 0.0

    def home(self, wait=True):
        near_rest = np.max(np.abs(np.asarray(self.read_q()) - REST_Q)) < np.radians(25)
        if near_rest:
            self.move_q(REST_VIA_Q, wait=True)
        self.move_q(HOME_Q, wait=wait)

    def rest(self, wait=True):
        """As omx.OMX.rest(): home, close an open gripper, REST_VIA_Q, REST_Q."""
        if np.max(np.abs(np.asarray(self.read_q()) - REST_Q)) < np.radians(5):
            return
        self.move_q(HOME_Q, wait=True)
        if self.read_extra()[1] > 0.2:
            self.set_gripper('closed')
        self.move_q(REST_VIA_Q, wait=True)
        self.move_q(REST_Q, wait=wait)

    def close(self):
        self.stop_recording()
        if self.viewer is not None:
            self.viewer.close()

    # ---- video export -------------------------------------------------------
    def record(self, path, fps=25, size=(640, 360), scene='free', lookat=(0.20, 0.0, 0.10), distance=0.8,
               azimuth=-35.0, elevation=-20.0, label=True):
        """Start an MP4 (mp4v) showing the wrist camera and a scene camera SIDE BY SIDE, each
        rendered at `size` (w, h). One frame is taken every 1/fps s of simulated time inside step(), so move_q(wait=True),
        stream() and settle() all record themselves; set_q() teleports do not (call snapshot()).
        scene: 'free' = a fixed viewpoint given by lookat/distance/azimuth/elevation (MuJoCo free-camera convention),
        or the name of a model camera such as 'overhead'. Stop with stop_recording() or close()."""
        import cv2
        self.stop_recording()
        W, H = int(size[0]), int(size[1])
        every = max(1, int(round(1.0 / (fps * self.dt))))
        fps_eff = 1.0 / (every * self.dt)
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps_eff, (2 * W, H))   # MPEG-4 part 2: plays everywhere, no codec install
        if not writer.isOpened():
            raise IOError(f'could not open a video writer for {path}')
        cam = scene
        if scene == 'free':
            cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
        self._rec = dict(path=path, writer=writer, renderer=mujoco.Renderer(self.model, height=H, width=W), size=(W, H),
                         scene=cam, every=every, k=0, frames=0, fps=fps_eff, label=label)
        self.snapshot()
        return path

    def snapshot(self):
        """Append one frame (wrist | scene) to the video started with record()."""
        if self._rec is None:
            return
        import cv2
        r = self._rec['renderer']; W, H = self._rec['size']
        r.update_scene(self.data, camera='wrist'); wrist = r.render()[..., ::-1].copy()
        r.update_scene(self.data, camera=self._rec['scene']); scene = r.render()[..., ::-1].copy()
        if self._rec['label']:
            t = self.data.time
            for img, txt in ((wrist, 'wrist camera'), (scene, 'scene')):
                cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(scene, f't = {t:6.2f} s', (W - 130, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        frame = np.hstack([wrist, scene])
        frame[:, W - 1:W + 1] = 255                        # thin divider
        self._rec['writer'].write(frame); self._rec['frames'] += 1

    def stop_recording(self):
        """Finish the video. Returns (path, number of frames, fps) or None if nothing was recording."""
        if self._rec is None:
            return None
        rec = self._rec; self._rec = None
        rec['writer'].release(); rec['renderer'].close()
        return rec['path'], rec['frames'], rec['fps']

    # ---- twin-only extras ---------------------------------------------------
    def set_q(self, q_reported):
        """Teleport (no dynamics) - handy for kinematic checks."""
        self.data.qpos[:4] = np.asarray(q_reported, float) + self.off
        self.data.qvel[:4] = 0; self.q_ref = self.q_goal = self.data.qpos[:4].copy()
        mujoco.mj_forward(self.model, self.data)

    def gravity_torque(self, q_reported=None):
        """g(q) from the physics model (qfrc_bias at zero velocity) - compare with Lab 7 identification."""
        if q_reported is not None:
            self.set_q(q_reported)
        self.data.qvel[:] = 0; mujoco.mj_forward(self.model, self.data)
        return self.data.qfrc_bias[:4].copy()

    def inertia(self, q_reported=None):
        if q_reported is not None:
            self.set_q(q_reported)
        M = np.zeros((self.model.nv, self.model.nv)); mujoco.mj_fullM(self.model, M, self.data.qM)
        return M[:4, :4]

    def true_T_BE(self):
        s = self.data.site('ee'); return _T(s.xmat.reshape(3, 3), s.xpos)

    def true_T_BC(self):
        """True camera pose (OpenCV convention) in the base frame."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist')
        R_mj = self.data.cam_xmat[cid].reshape(3, 3)
        return _T(R_mj @ np.diag([1, -1, -1]), self.data.cam_xpos[cid])

    def true_T_CM(self):
        return np.linalg.inv(self.true_T_BC()) @ self.mat_T

    def ground_truth(self):
        return dict(kin=self.kin, joint_offsets=self.off, T_EC=self.T_EC_true, T_BM=self.mat_T,
                    objects=self.objects.truth() if self.objects else {})

    def render(self, camera='wrist'):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.cam_res[1], width=self.cam_res[0])
        self._renderer.update_scene(self.data, camera=camera)
        rgb = self._renderer.render()
        return rgb[..., ::-1].copy()          # BGR for OpenCV

    def camera_K(self):
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'wrist')
        fovy = self.model.cam_fovy[cid]; W, H = self.cam_res
        f = 0.5 * H / np.tan(0.5 * np.radians(fovy))
        return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]]), np.zeros(5)


JOINT_INDEX = {11: 0, 12: 1, 13: 2, 14: 3}


def load_urdf_variant(urdf_path):
    """When the ROBOTIS URDF + meshes are available: MuJoCo compiles URDF directly. Returns an MjModel.
    You will then need to add the work cell, the wrist camera and actuators (see build_mjcf for the pattern),
    e.g. by editing the MjSpec: spec = mujoco.MjSpec.from_file(urdf_path); ...; spec.compile()."""
    return mujoco.MjModel.from_xml_path(urdf_path)


# ------------------------------------------------------------------------------
try:
    from omx_camera import Camera as _Camera
    import cv2

    class SimCamera(_Camera):
        """omx_camera.Camera driven by the twin's rendered wrist view. K is exact (no distortion); students may
        still run calibrate_from_mat() and compare with true_K()."""

        def __init__(self, sim, use_true_K=True):
            self.sim = sim
            self.K, self.dist = sim.camera_K() if use_true_K else (None, None)
            self.meta, self.corners_M = __import__('omx_camera').load_mat(scale=1.0)   # the twin renders the nominal mat
            self.adict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.meta['dictionary']))
            params = cv2.aruco.DetectorParameters()
            params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detector = cv2.aruco.ArucoDetector(self.adict, params)

        def read(self, flush=0):
            return self.sim.render('wrist')

        def true_K(self):
            return self.sim.camera_K()

        def show(self, T_EC=None):
            frame = self.read(); corners, ids = self.detect(frame)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids.reshape(-1, 1) if len(ids) else None)
            cv2.imshow('sim wrist camera', frame); cv2.waitKey(1)

        def close(self):
            pass
except ImportError:
    pass
