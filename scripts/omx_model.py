"""
omx_model.py - kinematic model of the OpenMANIPULATOR-X follower (OMX-F, 2025 leader/follower kit) for the
Robotics Toolbox for Python (roboticstoolbox-python) and spatialmath-python.

Nominal geometry (metres) from the ROBOTIS MuJoCo menagerie model `robotis_omx/omx.xml` (body offsets of the
`follower_*` chain; see labs/assets/omx_f). The arm has five joints plus a gripper; joint 5 (wrist roll about the
tool axis, ID 15) is LOCKED at zero in this module and the whole course treats the arm as 4-DOF
(yaw, shoulder pitch, elbow pitch, wrist pitch). Joint zero: link 2 vertical (with a 41.5 mm forward step),
links 3 and 4 horizontal. The end-effector frame {E} is at the closed fingertips (123 mm from the joint-4 axis:
28.7 mm to the roll axis, 29.5 mm to the finger root, 65 mm finger). If you fit a pen holder, change L4.

Students refine these values in the assignment (calibration).
"""
import numpy as np
import roboticstoolbox as rtb
from roboticstoolbox import ET
from spatialmath import SE3

# nominal link parameters (m) - VERIFY on the real arm
D1 = 0.0975    # mounting plane (mat surface) to joint-2 axis: 0.034 (base to J1 flange) + 0.0635 (J1 to J2)
L2Z = 0.11315  # joint 2 to joint 3, vertical component at q=0
L2X = 0.0415   # joint 2 to joint 3, forward component at q=0
L3 = 0.162     # joint 3 to joint 4
L4 = 0.123     # joint 4 to end effector (closed fingertips); roll axis (locked joint 5) is at 0.0287

# Home pose (rad): tool tip ~191 mm out and 129 mm up, pitched 30 deg below horizontal. Set by hand on the first arm and read
# with torque off (30 Sep 2026: 0.5, -58.2, 60.8, 32.4 deg), rounded to 5 deg. q = 0 is the kinematic ZERO pose (link 2
# vertical, links 3-4 horizontal), not home.
HOME_Q = np.radians([0.0, -60.0, 60.0, 30.0])
# Rest (parking) pose (rad): placed by hand just clear of the table and base on the first arm and read with torque off
# (30 Sep 2026: -28.0, -54.5, 94.1, -0.8 deg), rounded 5 deg AWAY from contact: tip ~22 mm above the table. With torque off
# the arm then settles the last few mm; nothing is pressed (the earlier (-30, -50, 85, 80) pose drove the gripper into
# the table and base and tripped its voltage protection). Reached via REST_VIA_Q (home with the wrist already up): with
# velocity-based profiles the joints arrive at different times, and a direct home -> rest move could put the tip 23 mm
# below the table if joints 2-3 finished before the wrist; via the waypoint the tip stays >= 22 mm for any joint order.
REST_Q = np.radians([-30.0, -55.0, 95.0, -5.0])
REST_VIA_Q = np.radians([0.0, -60.0, 60.0, -5.0])

# --- wrist camera and tool offsets -------------------------------------------
# Camera pose in the end-effector frame {E} (x along link 4), from CAD - VERIFY, refined in Lab 9.
# Camera axes in {E}: optical axis z_C = +x_E (looks along the tool), x_C = -y_E, y_C = -z_E.
R_EC = np.array([[0, 0, 1],
                 [-1, 0, 0],
                 [0, -1, 0]])             # columns = camera axes (x_C, y_C, z_C) expressed in {E}
T_EC = SE3.Rt(R_EC, [-0.030, 0.0, 0.040])  # camera 30 mm behind and 40 mm above the EE point - a CAD PLACEHOLDER: real mounts differ
                                           # (first arm: optical axis ~62 deg below the tool); Lab 2 measures yours


def mount_tilt(T_EC):
    """Angle (rad) of the camera's optical axis below the tool axis x_E for a mount T_EC (SE3 or 4x4); 0 for the CAD placeholder."""
    z = np.asarray(getattr(T_EC, 'A', T_EC))[:3, 2]
    return float(np.arctan2(-z[2], z[0]))


def look_down_pitch(T_EC, view_pitch=np.pi / 2):
    """Tool pitch (rad, positive = nose down, as omx_ik takes it) that points the camera's optical axis view_pitch below the
    horizontal (pi/2 = straight down at the mat) for the mount T_EC: view_pitch - mount_tilt(T_EC)."""
    return view_pitch - mount_tilt(T_EC)


P_TIP_E = np.array([0.0, -0.0016, 0.0])    # tool tip in {E}: the finger centre is 1.6 mm to -y (ROBOTIS omx_f.urdf: the gripper
                                           # pivots are at y +7.5 / -10.8 mm); pen holder: e.g. [0.055, -0.0016, 0]

# OMX-F ranges: J2 -107..+95 and J3 -90..+96 deg (ROBOTIS omx_f.ros2_control.xacro, ticks 830-3129 / 1024-3140), J4 +/-100 deg
# (e-manual). Joint 1 (spec -270..+360) reduced to +/-150 deg for the camera cable.
Q_LIM = np.array([[-np.radians(150), np.radians(150)],
                  [-1.868, 1.658],
                  [-1.571, 1.676],
                  [-1.745, 1.745]])


def omx_ets(d1=D1, l2z=L2Z, l2x=L2X, l3=L3, l4=L4):
    """Elementary Transform Sequence for the OMX."""
    return (ET.tz(d1) * ET.Rz()             # joint 1: base yaw
            * ET.Ry() * ET.tz(l2z) * ET.tx(l2x)  # joint 2: shoulder pitch
            * ET.Ry() * ET.tx(l3)           # joint 3: elbow pitch
            * ET.Ry() * ET.tx(l4))          # joint 4: wrist pitch


def omx_robot(**kw):
    """Robotics Toolbox Robot object built from the ETS."""
    robot = rtb.Robot(omx_ets(**kw), name='OMX')
    robot.qlim = Q_LIM.T
    robot.addconfiguration('zero', np.zeros(4))
    robot.addconfiguration('home', HOME_Q)
    robot.addconfiguration('rest', REST_Q)
    robot.addconfiguration('ready', np.array([0, 0.6, -0.4, 0.4]))
    return robot


def omx_dh():
    """Standard DH version of the same arm (for comparison with the ETS model).

    The shoulder link has an in-plane offset (l2x), handled by a link of length
    sqrt(l2x^2 + l2z^2) with a fixed joint-angle offset, and a compensating
    offset on the next joint.
    """
    a2 = np.hypot(L2X, L2Z)
    beta = np.arctan2(L2X, L2Z)           # angle of link 2 from vertical
    links = [
        rtb.RevoluteDH(d=D1, a=0, alpha=-np.pi / 2, qlim=Q_LIM[0]),   # -pi/2 so z1 = +y0 (nose-down positive)
        rtb.RevoluteDH(d=0, a=a2, alpha=0, offset=beta - np.pi / 2, qlim=Q_LIM[1]),
        rtb.RevoluteDH(d=0, a=L3, alpha=0, offset=np.pi / 2 - beta, qlim=Q_LIM[2]),
        rtb.RevoluteDH(d=0, a=L4, alpha=0, qlim=Q_LIM[3]),
    ]
    return rtb.DHRobot(links, name='OMX-DH')


def omx_ik(p, pitch=0.0, elbow='up', d1=D1, l2z=L2Z, l2x=L2X, l3=L3, l4=L4):
    """Analytic inverse kinematics for the OMX.

    p      : (x, y, z) end-effector position in the base frame (m)
    pitch  : desired pitch of the final link measured from horizontal,
             positive = pointing downwards (rad). 0 = pointing forwards.
    elbow  : 'up' (elbow above the shoulder-wrist line; the OMX's normal posture), 'down', or 'any'
             (try 'up' first, then 'down')
    Returns q (4,) in rad, or None if unreachable within the joint limits.

    Method: joint 1 is the azimuth of p. In the vertical plane containing the
    arm, the wrist centre is found by backing off l4 along the pitch direction,
    then a planar 2R problem (link lengths a2 and l3) is solved for joints 2-3,
    with joint 4 making up the required pitch.
    """
    x, y, z = p
    q1 = np.arctan2(y, x)
    r = np.hypot(x, y)               # radial distance in the horizontal plane
    # wrist centre in the (r, z) plane
    rw = r - l4 * np.cos(pitch)
    zw = z - d1 + l4 * np.sin(pitch)
    a2 = np.hypot(l2x, l2z)
    beta = np.arctan2(l2x, l2z)
    dd = rw ** 2 + zw ** 2
    c3 = (dd - a2 ** 2 - l3 ** 2) / (2 * a2 * l3)
    if abs(c3) > 1:
        return None
    if elbow == 'any':
        q = omx_ik(p, pitch, 'up', d1, l2z, l2x, l3, l4)
        return q if q is not None else omx_ik(p, pitch, 'down', d1, l2z, l2x, l3, l4)
    s3 = np.sqrt(1 - c3 ** 2) * (-1 if elbow == 'up' else 1)   # elbow-up = negative interior angle in this convention
    phi3 = np.arctan2(s3, c3)                       # interior angle between link 2 and link 3
    # angle of link 2 measured from horizontal (in the r-z plane)
    phi2 = np.arctan2(zw, rw) - np.arctan2(l3 * s3, a2 + l3 * c3)
    # convert to OMX joint angles: q2 = 0 -> link 2 along +z rotated by beta towards +r
    q2 = (np.pi / 2 - beta) - phi2
    # relative (interior) angle phi3 is measured anticlockwise (up); OMX q3 is nose-down positive
    q3 = (beta - np.pi / 2) - phi3   # q3 = 0 gives link 3 horizontal when q2 = 0
    # wrist: total pitch of link 4 below horizontal = q2 + q3 + q4 (Ry positive = nose down)
    q4 = pitch - (q2 + q3)
    q = np.array([q1, q2, q3, q4])
    if np.any(q < Q_LIM[:, 0] - 1e-9) or np.any(q > Q_LIM[:, 1] + 1e-9):
        return None
    return q


def fk_check(robot=None, n=200, seed=0):
    """Round-trip test: fkine -> omx_ik -> fkine. Returns max position error (m)."""
    rng = np.random.default_rng(seed)
    robot = robot or omx_robot()
    worst = 0.0
    for _ in range(n):
        q = rng.uniform(Q_LIM[:, 0], Q_LIM[:, 1])
        T = robot.fkine(q)
        pitch = q[1] + q[2] + q[3]
        for elbow in ('up', 'down'):
            qs = omx_ik(T.t, pitch, elbow)
            if qs is not None:
                err = np.linalg.norm(robot.fkine(qs).t - T.t)
                worst = max(worst, err)
    return worst


if __name__ == '__main__':
    r = omx_robot()
    print(r)
    print('zero pose (q = 0):\n', r.fkine(np.zeros(4)))
    print('IK round-trip worst-case error (m):', fk_check(r))
