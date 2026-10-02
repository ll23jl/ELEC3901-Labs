"""
omx.py - minimal Python interface to the ROBOTIS OpenMANIPULATOR-X follower (OMX-F)
for ELEC3901 Robotic Manipulators.

Hardware (ROBOTIS OMX-F, 2025 leader/follower kit, e-manual "hardware" page):
  * joints 1-3: Dynamixel XL430-W250-T, IDs 11, 12, 13   (position/velocity/PWM modes only; reports Present LOAD, not current)
  * joints 4-5 and gripper: XL330-M288-T, IDs 14, 15, 16  (current sensing and current-based position mode available)
  * joint 5 (ID 15, wrist roll about the tool axis) is LOCKED at its zero by this wrapper: the module treats the arm as
    4-DOF. read_q()/move_q() work with joints 1-4 only.
  * OpenRB-150 controller, USB-C, Protocol 2.0, 1 Mbaud, 12 V
  * joint zero (q = 0) corresponds to 2048 ticks (the kinematic zero pose; home() goes to HOME_Q instead)

IMPORTANT: this module has NOT yet been validated on the departmental arms.
Demonstrators must check joint directions, zero offsets and limits before
week 1 and adjust SIGN / OFFSET_TICKS / Q_LIM below.

Usage:
    from omx import OMX
    arm = OMX('/dev/ttyUSB0')        # Windows: 'COM3'
    arm.torque(True)
    print(arm.read_q())              # radians, shape (4,)
    arm.move_q([0, 0, 0, 0], wait=True)
    arm.gripper(open_=True)
    arm.close()

A software-only stand-in is available for working away from the lab:
    arm = OMX(simulate=True)
"""
import time
import numpy as np

try:
    from dynamixel_sdk import (PortHandler, PacketHandler, GroupSyncRead,
                               GroupSyncWrite, COMM_SUCCESS)
    _SDK = True
except ImportError:  # allow import on machines without the SDK
    _SDK = False

# ---- Dynamixel X-series control table, XL430-W250 + XL330-M288 (Protocol 2.0) ----
ADDR_DRIVE_MODE = 10        # 1 byte (bit 2: time-based profile)
ADDR_OPERATING_MODE = 11    # 1 byte
ADDR_HOMING_OFFSET = 20     # 4 bytes
ADDR_PWM_LIMIT = 36         # 2 bytes (0..885 = 0..100 %)
ADDR_TORQUE_ENABLE = 64     # 1 byte
ADDR_HW_ERROR = 70          # 1 byte, Hardware Error Status
ADDR_POS_D_GAIN = 80        # 2 bytes
ADDR_POS_I_GAIN = 82        # 2 bytes
ADDR_POS_P_GAIN = 84        # 2 bytes
ADDR_CURRENT_LIMIT = 38     # 2 bytes, EEPROM (XL330: 1 mA units)
ADDR_GOAL_CURRENT = 102     # 2 bytes
ADDR_GOAL_VELOCITY = 104    # 4 bytes
ADDR_PROFILE_ACCEL = 108    # 4 bytes
ADDR_PROFILE_VELOCITY = 112  # 4 bytes
ADDR_GOAL_POSITION = 116    # 4 bytes
COMM_RETRIES = 3            # extra attempts for a single register read/write that comes back garbled (seen 29-30 Sep)
ADDR_PRESENT_CURRENT = 126  # 2 bytes (signed)
ADDR_PRESENT_VELOCITY = 128  # 4 bytes (signed)
ADDR_PRESENT_POSITION = 132  # 4 bytes

MODE_CURRENT = 0
MODE_VELOCITY = 1
MODE_POSITION = 3
MODE_CURRENT_POSITION = 5   # current-based position control (torque-limited)

TICKS_PER_REV = 4096
RAD_PER_TICK = 2 * np.pi / TICKS_PER_REV
VEL_UNIT_RAD_S = 0.229 * 2 * np.pi / 60   # rad/s per raw unit (both servo types)
CURRENT_UNIT_A = 1.0e-3         # XL330: Present/Goal Current unit is 1 mA
LOAD_UNIT = 1.0e-3              # XL430: Present Load unit is 0.1 % of maximum (signed)
# stall figures from the e-manual: XL430-W250 1.4 N m at 1.3 A (11.1 V); XL330-M288 0.52 N m at 1.47 A (5 V)
STALL_TORQUE = np.array([1.4, 1.4, 1.4, 0.52])     # N m per joint 1..4
STALL_CURRENT_A = np.array([1.3, 1.3, 1.3, 1.47])
TORQUE_PER_AMP = STALL_TORQUE / STALL_CURRENT_A    # N m per A, per joint (approx)

JOINT_IDS = [11, 12, 13, 14]
ROLL_ID = 15                    # joint 5, held at ROLL_TICKS while torque is on
ROLL_TICKS = 2048
GRIPPER_ID = 16
XL330_IDS = {14, 15, 16}        # the only servos with current sensing / current-based position mode
# Home pose (rad): tool tip ~191 mm out and 129 mm up, pitched 30 deg below horizontal. Set by hand on the first arm and read
# with torque off (30 Sep 2026: 0.5, -58.2, 60.8, 32.4 deg), rounded to 5 deg. q = 0 is the kinematic ZERO pose (link 2
# vertical, links 3-4 horizontal), not home.
HOME_Q = np.radians([0.0, -60.0, 60.0, 30.0])   # = omx_model.HOME_Q
# Rest (parking) pose (rad): placed by hand just clear of the table and base on the first arm and read with torque off
# (30 Sep 2026: -28.0, -54.5, 94.1, -0.8 deg), rounded 5 deg AWAY from contact: tip ~22 mm above the table. With torque off
# the arm then settles the last few mm; nothing is pressed (the earlier (-30, -50, 85, 80) pose drove the gripper into
# the table and base and tripped its voltage protection). Reached via REST_VIA_Q (home with the wrist already up): with
# velocity-based profiles the joints arrive at different times, and a direct home -> rest move could put the tip 23 mm
# below the table if joints 2-3 finished before the wrist; via the waypoint the tip stays >= 22 mm for any joint order.
REST_Q = np.radians([-30.0, -55.0, 95.0, -5.0])   # = omx_model.REST_Q
REST_VIA_Q = np.radians([0.0, -60.0, 60.0, -5.0])   # = omx_model.REST_VIA_Q
HW_ERROR_BITS = {0: 'input voltage', 2: 'overheating', 3: 'motor encoder', 4: 'electrical shock', 5: 'overload'}
ALL_SERVO_IDS = JOINT_IDS + [ROLL_ID, GRIPPER_ID]
ARM_SERVO_IDS = JOINT_IDS + [ROLL_ID]   # velocity-based profiles and the selectable gains; the gripper keeps its own set-up
# Position P/I/D (Dynamixel units). 'standard' = ROBOTIS OMX-F ros2_control config (omx_f.ros2_control.xacro);
# 'precision' = ROBOTIS OMX drawing tuning (I removes steady-state error and friction, D damps overshoot).
STANDARD = 'standard'
GAIN_SETS = {'standard': (1000, 0, 1000), 'precision': (1000, 1000, 1200)}
SIGN = np.array([1, 1, 1, 1])            # flip if a joint moves the wrong way
OFFSET_TICKS = np.array([2048, 2048, 2048, 2048])
# Joint limits (rad): ROBOTIS OMX-F ros2_control limits J2 -107..+95 deg (ticks 830-3129), J3 -90..+96 deg (1024-3140);
# J4 +/-100 deg (e-manual); J1 (spec -270..+360) reduced to +/-150 deg
Q_LIM = np.array([[-2.618, 2.618],           # +/-150 deg: protects the wrist-camera USB cable
                  [-1.868, 1.658],               # J2 -107..+95 deg
                  [-1.571, 1.676],               # J3 -90..+96 deg
                  [-1.745, 1.745]])
# Gripper (ID 16) positions, measured on the first arm (30 Sep 2026): closed stop ~2005-2044 ticks, open stop 3220. Both targets
# stay short of the stops so the gripper never drives into one. For camera work the gripper stays OPEN (closed fingers hide the
# mat): the camera cell opens it at home, and only rest() closes it (folded at rest an open gripper hits the base).
GRIPPER_OPEN_TICKS = 3150
GRIPPER_CLOSE_TICKS = 2100
# Gripper speed profile (velocity-based units, as the joints): it had none (profile 0 = unlimited), so every open/close slammed
# into a stop in < 0.3 s at the 600 mA limit and the stop tripped all three XL330s (14, 15, 16) on 'input voltage'.
GRIPPER_PROFILE = (30, 10)
# Wrist roll (joint 5, ID 15): masked (held at ROLL_TICKS = 0 rad) unless commanded; limited to +/-90 deg for the camera cable.
ROLL_LIM = np.radians([-90.0, 90.0])
# Gripper as ROBOTIS configure the OMX-F (omx_f.ros2_control.xacro): current-based position mode, current limit and goal current
# 600 (x 1 mA = 0.6 A). It then stops pushing at 0.6 A instead of stalling at full current against a stop or an object (a stall
# on the first arm, 30 Sep 2026, latched the XL330's 'input voltage' error).
GRIPPER_MODE = 5
GRIPPER_CURRENT_LIMIT = 600


ROBOTIS_USB_VID = 0x2F5D           # OpenRB-150 / U2D2


def find_port():
    """Serial port of the OpenRB-150 (ROBOTIS USB vendor id). Raises if none or more than one is connected."""
    from serial.tools import list_ports
    ports = [p.device for p in list_ports.comports() if p.vid == ROBOTIS_USB_VID]
    if len(ports) != 1:
        others = [f'{p.device} ({p.description})' for p in list_ports.comports()]
        raise IOError(f'{len(ports)} ROBOTIS controllers found ({ports}); set PORT by hand. Serial ports: {others}')
    return ports[0]


def _s16(v):
    return v - 65536 if v > 32767 else v


def _s32(v):
    return v - 4294967296 if v > 2147483647 else v


class OMX:
    """Thin wrapper around the Dynamixel SDK for the OpenMANIPULATOR-X."""

    def __init__(self, port='auto', baud=1_000_000, simulate=False, gains='standard'):
        """port: 'auto' finds the OpenRB-150 by its ROBOTIS USB vendor id (find_port); or give it, e.g. 'COM4'.
        gains: 'standard' (ROBOTIS OMX-F, P/I/D 1000/0/1000) or 'precision' (1000/1000/1200: the I term removes the 1-4 deg
        a gravity-loaded joint otherwise stops short by; it made the first arm shake, 30 Sep 2026), see GAIN_SETS. On connect every arm servo (IDs 11-15) is put on a
        velocity-based profile (drive mode bit 2 cleared), so set_profile() takes speeds, never move times; the gripper
        (ID 16) keeps its own configuration."""
        self.simulate = simulate or not _SDK
        self.port_name = 'simulated'
        self.gains = gains
        self._alerted = set()
        self._q_sim = np.zeros(4)
        self._grip_sim = GRIPPER_OPEN_TICKS
        if self.simulate:
            if not _SDK and not simulate:
                print('dynamixel_sdk not found - running in SIMULATE mode')
            return
        if port == 'auto':
            port = find_port()
        self.port_name = port
        self.port = PortHandler(port)
        self.pkt = PacketHandler(2.0)
        if not self.port.openPort():
            raise IOError(f'Could not open {port}')
        if not self.port.setBaudRate(baud):
            self.port.closePort()          # release the port, or it stays locked until the kernel restarts
            raise IOError('Could not set baud rate')
        self._sr = GroupSyncRead(self.port, self.pkt, ADDR_PRESENT_POSITION, 4)
        for i in JOINT_IDS:
            self._sr.addParam(i)
        self._sw = GroupSyncWrite(self.port, self.pkt, ADDR_GOAL_POSITION, 4)
        self._alerted = set()
        try:
            self._clear_hardware_errors()
            self._setup_gripper()
            self._use_velocity_profiles()
            self.set_gains(STANDARD, ids=[GRIPPER_ID])           # gripper: ROBOTIS standard gains, never precision
            self.set_gains(gains)
        except Exception:
            self.port.closePort()                                # do not leave the port locked
            raise

    def _clear_hardware_errors(self):
        """On connect: report servo hardware errors; reboot the flagged servos that have torque off (safe: nothing is held),
        refuse if a flagged servo is holding the arm. First seen 30 Sep 2026: the gripper (XL330, 6.0 V bus, 3.5-7.0 V limit)
        latched 'input voltage' while closing onto its stop."""
        errs = self.hardware_errors()
        if not errs:
            return
        print('servo hardware errors:', errs)
        held = [i for i in errs if self._read(i, ADDR_TORQUE_ENABLE, 1)]
        if held:
            raise IOError(f'servos {held} have a hardware error and torque on. Support the arm, power-cycle it, then connect again.')
        self.reboot(list(errs))
        print('rebooted', list(errs), '- errors cleared')

    def _setup_gripper(self):
        """Gripper to current-based position mode with a 600 mA limit (see GRIPPER_MODE). Mode and limit are EEPROM: they are
        written only while the gripper's torque is off. With torque on it may be holding something (the pen), so it is left
        alone with a warning; the goal current (RAM) is always set."""
        mode = self._read(GRIPPER_ID, ADDR_OPERATING_MODE, 1)
        limit = self._read(GRIPPER_ID, ADDR_CURRENT_LIMIT, 2)
        if (mode, limit) != (GRIPPER_MODE, GRIPPER_CURRENT_LIMIT):
            if self._read(GRIPPER_ID, ADDR_TORQUE_ENABLE, 1):
                print(f'WARNING: gripper is in mode {mode} (current limit {limit}) with torque on; not reconfigured. Empty it, '
                      'switch its torque off and reconnect to get the current-limited mode.')
            else:
                self._write(GRIPPER_ID, ADDR_OPERATING_MODE, 1, GRIPPER_MODE)
                self._write(GRIPPER_ID, ADDR_CURRENT_LIMIT, 2, GRIPPER_CURRENT_LIMIT)
                got = (self._read(GRIPPER_ID, ADDR_OPERATING_MODE, 1), self._read(GRIPPER_ID, ADDR_CURRENT_LIMIT, 2))
                if got != (GRIPPER_MODE, GRIPPER_CURRENT_LIMIT):
                    raise IOError(f'gripper: wrote mode {GRIPPER_MODE} / limit {GRIPPER_CURRENT_LIMIT}, read back {got}')
                print(f'gripper: mode {mode} -> {GRIPPER_MODE} (current-based position), current limit {limit} -> {GRIPPER_CURRENT_LIMIT} mA')
        self._write(GRIPPER_ID, ADDR_GOAL_CURRENT, 2, GRIPPER_CURRENT_LIMIT)
        self.set_profile(*GRIPPER_PROFILE, ids=[GRIPPER_ID])    # RAM: gentle open/close (see GRIPPER_PROFILE)

    def _use_velocity_profiles(self):
        """Clear drive-mode bit 2 (time-based profile) on the arm servos, keeping the other bits (e.g. reverse direction).
        ROBOTIS configure the OMX-F time-based (profile registers = move time in ms), which would make set_profile(60, 20)
        mean 'every move in 60 ms'. Drive mode is EEPROM: it can only be written with torque off, so a servo that is
        time-based with torque ON is refused rather than switched off (the arm would drop)."""
        for i in ARM_SERVO_IDS:
            dm = self._read(i, ADDR_DRIVE_MODE, 1)
            if dm & 0x04:
                if self._read(i, ADDR_TORQUE_ENABLE, 1):
                    raise IOError(f'servo {i} uses a time-based profile (drive mode {dm}) with torque ON. Support the arm, '
                                  'switch torque off (or power-cycle it), then connect again.')
                self._write(i, ADDR_DRIVE_MODE, 1, dm & ~0x04)
                if self._read(i, ADDR_DRIVE_MODE, 1) & 0x04:
                    raise IOError(f'servo {i}: could not switch to a velocity-based profile')
                print(f'servo {i}: drive mode {dm} -> {dm & ~0x04} (velocity-based profile)')

    def set_gains(self, name='standard', ids=None):
        """Position P/I/D from GAIN_SETS ('standard' or 'precision') on the arm servos (IDs 11-15; the gripper only ever
        gets 'standard'). Gains are RAM registers: they can change with torque on. Written values are read back."""
        P, I, D = GAIN_SETS[name]
        ids = ids or ARM_SERVO_IDS
        if name != STANDARD and GRIPPER_ID in ids:
            raise ValueError('the gripper keeps the standard gains (an I term keeps building current while it grips)')
        for i in ids:
            for addr, v in ((ADDR_POS_D_GAIN, D), (ADDR_POS_I_GAIN, I), (ADDR_POS_P_GAIN, P)):
                self._write(i, addr, 2, v)
                if not self.simulate and self._read(i, addr, 2) != v:
                    raise IOError(f'servo {i}: gain register {addr} did not read back {v}')
        if ids is ARM_SERVO_IDS or set(ARM_SERVO_IDS) <= set(ids):
            self.gains = name

    # ---- low level -------------------------------------------------------
    def _write(self, dxl_id, addr, nbytes, value):
        if self.simulate:
            return
        f = {1: self.pkt.write1ByteTxRx, 2: self.pkt.write2ByteTxRx,
             4: self.pkt.write4ByteTxRx}[nbytes]
        value = int(value) & ((1 << (8 * nbytes)) - 1)
        for _ in range(COMM_RETRIES + 1):                         # a garbled packet on the shared bus is retried (writes are idempotent)
            res, err = f(self.port, dxl_id, addr, value)
            if res == COMM_SUCCESS:
                break
        else:
            raise IOError(f'servo {dxl_id}: {self.pkt.getTxRxResult(res)}')
        if err & 0x7F:                                           # a real packet error: the command was not carried out
            raise IOError(f'servo {dxl_id}: {self.pkt.getRxPacketError(err)}')
        if err & 0x80:                                           # alert bit only: done, but the servo has a hardware error
            self._alert(dxl_id)

    def _read(self, dxl_id, addr, nbytes):
        if self.simulate:
            return 0
        f = {1: self.pkt.read1ByteTxRx, 2: self.pkt.read2ByteTxRx,
             4: self.pkt.read4ByteTxRx}[nbytes]
        for _ in range(COMM_RETRIES + 1):
            val, res, err = f(self.port, dxl_id, addr)
            if res == COMM_SUCCESS:
                break
        else:
            raise IOError(f'servo {dxl_id}: {self.pkt.getTxRxResult(res)}')
        if err & 0x80 and addr != ADDR_HW_ERROR:
            self._alert(dxl_id)
        return val

    def _alert(self, dxl_id):
        """Warn once per servo that its Hardware Error Status is set (see hardware_errors(), reboot())."""
        if dxl_id not in self._alerted:
            self._alerted.add(dxl_id)
            print(f'WARNING: servo {dxl_id} reports a hardware error: {self.hardware_errors().get(dxl_id, "?")}. '
                  f'Support the arm, torque off, then arm.reboot([{dxl_id}]) or power-cycle.')

    def hardware_errors(self, ids=None):
        """{id: [names]} for every servo whose Hardware Error Status is set (empty dict = all clear)."""
        out = {}
        for i in (ids or ALL_SERVO_IDS):
            hw = self._read(i, ADDR_HW_ERROR, 1)
            if hw:
                out[i] = [n for b, n in HW_ERROR_BITS.items() if hw & (1 << b)] or [f'code {hw}']
        return out

    def recover(self, ids=None):
        """Bring tripped servos back: those with a hardware error AND torque off are rebooted, their RAM set-up (gains,
        profile, gripper goal current) re-applied and torque switched back on. A tripped servo holding the arm is refused
        (see reboot()). Returns the list of servos recovered."""
        bad = list(self.hardware_errors(ids))
        if not bad:
            return []
        print('recovering servos', bad, 'from', {i: self.hardware_errors([i])[i] for i in bad})
        self.reboot(bad)
        arm_ids = [i for i in bad if i in ARM_SERVO_IDS]
        if GRIPPER_ID in bad:
            self._setup_gripper()
            self.set_gains(STANDARD, ids=[GRIPPER_ID])
        if arm_ids:
            self.set_gains(self.gains, ids=arm_ids)
            self.set_profile(*getattr(self, '_profile', (40, 10)), ids=arm_ids)
        self.torque(True, ids=bad)
        return bad

    def reboot(self, ids):
        """Reboot servos to clear a hardware error. Refused for a servo with torque on (it would go limp: the arm drops).
        A reboot restores the servo's RAM (gains, profile), so the connect-time set-up is re-applied afterwards."""
        for i in ids:
            if self._read(i, ADDR_TORQUE_ENABLE, 1):
                raise IOError(f'servo {i} has torque on: support the arm and switch torque off before rebooting it')
        for i in ids:
            self.pkt.reboot(self.port, i)
        time.sleep(0.6)                                          # the servos restart
        self._alerted -= set(ids)
        left = self.hardware_errors(ids)
        if left:
            raise IOError(f'hardware error still set after reboot: {left} - check the supply and the servo')

    # ---- configuration ---------------------------------------------------
    def torque(self, on, ids=None):
        for i in (ids or JOINT_IDS + [ROLL_ID, GRIPPER_ID]):
            self._write(i, ADDR_TORQUE_ENABLE, 1, 1 if on else 0)
        if on and (ids is None or ROLL_ID in ids):
            self._write(ROLL_ID, ADDR_GOAL_POSITION, 4, ROLL_TICKS)      # keep the wrist roll (joint 5) locked at zero

    def set_mode(self, mode, ids=None):
        """Operating mode must be changed with torque OFF. Current-based position mode (5) and current mode (0) exist only
        on the XL330 servos (IDs 14-16); the XL430 joints 1-3 are left in position mode (3) with a warning."""
        for i in (ids or JOINT_IDS):
            m = mode
            if mode in (MODE_CURRENT, MODE_CURRENT_POSITION) and i not in XL330_IDS:
                print(f'servo {i} is an XL430 (no current control): staying in position mode'); m = MODE_POSITION
            self._write(i, ADDR_TORQUE_ENABLE, 1, 0)
            self._write(i, ADDR_OPERATING_MODE, 1, m)

    def set_profile(self, velocity=60, acceleration=20, ids=None):
        """Profile velocity (0.229 rpm units) and acceleration (214.577 rev/min^2 units).
        Smaller = gentler. 0 = no limit (do NOT use 0 in the teaching lab). Default: joints 1-4 AND the wrist roll (ID 15),
        which otherwise has no limit and snaps round when commanded."""
        for i in (ids or ARM_SERVO_IDS):
            self._write(i, ADDR_PROFILE_VELOCITY, 4, velocity)
            self._write(i, ADDR_PROFILE_ACCEL, 4, acceleration)
        if ids is None:
            self._profile = (velocity, acceleration)            # re-applied by recover() after a reboot

    def set_pid(self, P=1000, I=0, D=1000, ids=None):
        """Raw position gains. Defaults = ROBOTIS standard; prefer set_gains('standard' | 'precision')."""
        for i in (ids or JOINT_IDS):
            self._write(i, ADDR_POS_P_GAIN, 2, P)
            self._write(i, ADDR_POS_I_GAIN, 2, I)
            self._write(i, ADDR_POS_D_GAIN, 2, D)

    def set_goal_current(self, amps, ids=None):
        """Torque limit / feedforward in current-based position mode (mode 5). XL330 only (IDs 14-16): on the XL430
        joints 1-3 the register does not exist and the call is ignored. Use set_pwm_limit() to soften those."""
        raw = int(np.clip(amps / CURRENT_UNIT_A, -1750, 1750))
        for i in (ids or JOINT_IDS):
            if i in XL330_IDS:
                self._write(i, ADDR_GOAL_CURRENT, 2, raw)

    def set_pwm_limit(self, fraction, ids=None):
        """Cap the servo drive (PWM Limit, address 36, 0..885 = 0..100 %) - the XL430's only torque-limiting knob.
        Change with torque OFF. fraction in (0, 1]."""
        raw = int(np.clip(fraction, 0.05, 1.0) * 885)
        for i in (ids or JOINT_IDS):
            self._write(i, ADDR_TORQUE_ENABLE, 1, 0)
            self._write(i, ADDR_PWM_LIMIT, 2, raw)

    # ---- conversions -----------------------------------------------------
    @staticmethod
    def ticks_to_rad(ticks):
        return SIGN * (np.asarray(ticks) - OFFSET_TICKS) * RAD_PER_TICK

    @staticmethod
    def rad_to_ticks(q):
        return np.round(OFFSET_TICKS + SIGN * np.asarray(q) / RAD_PER_TICK).astype(int)

    # ---- motion ----------------------------------------------------------
    def read_q(self):
        """Joint angles (rad) for joints 1..4."""
        if self.simulate:
            return self._q_sim.copy()
        for attempt in range(4):                     # an occasional packet is lost on the bus (seen 29 Sep): retry
            if self._sr.txRxPacket() == COMM_SUCCESS:
                break
            time.sleep(0.01)
        else:
            raise IOError('sync read failed 4 times - check the cable, the 12 V supply and each servo (ping)')
        ticks = [self._sr.getData(i, ADDR_PRESENT_POSITION, 4) for i in JOINT_IDS]
        return self.ticks_to_rad(ticks)

    def read_qd(self):
        """Joint velocities (rad/s)."""
        return np.array([SIGN[k] * _s32(self._read(i, ADDR_PRESENT_VELOCITY, 4)) * VEL_UNIT_RAD_S
                         for k, i in enumerate(JOINT_IDS)])

    def read_load(self):
        """Signed load fraction (-1..1) per joint: XL430 joints 1-3 report Present Load (0.1 % units of the maximum
        drive); the XL330 joint 4 reports Present Current, here divided by its stall current."""
        out = []
        for k, i in enumerate(JOINT_IDS):
            raw = _s16(self._read(i, ADDR_PRESENT_CURRENT, 2))       # address 126 is Present Load (XL430) or Present Current (XL330)
            out.append(SIGN[k] * (raw * CURRENT_UNIT_A / STALL_CURRENT_A[k] if i in XL330_IDS else raw * LOAD_UNIT))
        return np.array(out)

    def read_current(self):
        """Approximate motor current (A) per joint. Exact for the XL330 joint 4; for the XL430 joints 1-3 it is the load
        fraction scaled by the stall current (the XL430 has no current sensor). Multiply by TORQUE_PER_AMP for torque."""
        return self.read_load() * STALL_CURRENT_A

    def read_torque(self):
        """Approximate joint torque (N m) = load fraction x stall torque."""
        return self.read_load() * STALL_TORQUE

    def move_q(self, q, wait=False, tol=0.02, timeout=8.0, still_tol=0.003, still_s=0.3, roll=None, gripper=None):
        """Command joint angles (rad) of joints 1-4. Clips to Q_LIM. Optional: roll (rad, joint 5, clipped to ROLL_LIM) and
        gripper ('open', 'closed' or a fraction 0 closed .. 1 open). Left as None they are MASKED - the roll stays held at zero
        and the gripper where it is - which is what every lab uses. wait=True blocks until the joints are within tol (rad) of q, OR have
        stopped moving (< still_tol rad for still_s): with the standard gains (I = 0) a gravity-loaded joint stops 1-4 deg
        short and never gets within tol, which used to cost the full timeout on every waited move."""
        q = np.clip(np.asarray(q, dtype=float), Q_LIM[:, 0], Q_LIM[:, 1])
        if roll is not None:
            self.set_roll(roll)
        if gripper is not None:
            self.set_gripper(gripper)
        if self.simulate:
            self._q_sim = q
            return
        self._sw.clearParam()
        for i, t in zip(JOINT_IDS, self.rad_to_ticks(q)):
            t = int(t)
            self._sw.addParam(i, [t & 0xFF, (t >> 8) & 0xFF, (t >> 16) & 0xFF, (t >> 24) & 0xFF])
        if self._sw.txPacket() != COMM_SUCCESS:
            raise IOError('sync write failed')
        if wait:
            t0 = time.time(); q_prev = self.read_q(); t_still = None
            while time.time() - t0 < timeout:
                time.sleep(0.02)
                q_now = self.read_q()
                if np.max(np.abs(q_now - q)) < tol:
                    return
                if np.max(np.abs(q_now - q_prev)) < still_tol:
                    t_still = t_still or time.time()
                    if time.time() - t_still > still_s and time.time() - t0 > 0.5:
                        return                                   # stopped short (gravity, no I term): settled
                else:
                    t_still = None
                q_prev = q_now

    def stream(self, qs, dt):
        """Stream a trajectory (N x 4 array) at fixed period dt; returns logged (t, q_meas)."""
        log_t, log_q = [], []
        t0 = time.time()
        for k, q in enumerate(qs):
            self.move_q(q)
            log_t.append(time.time() - t0)
            log_q.append(self.read_q())
            sleep = t0 + (k + 1) * dt - time.time()
            if sleep > 0:
                time.sleep(sleep)
        return np.array(log_t), np.array(log_q)

    def gripper(self, open_=True):
        self.set_gripper('open' if open_ else 'closed')

    def set_gripper(self, g, wait=False, timeout=3.0):
        """Gripper to 'open', 'closed' or a fraction (0 closed .. 1 open) of the measured range. Torque must be on (ID 16)."""
        frac = {'open': 1.0, 'closed': 0.0}[g] if isinstance(g, str) else float(np.clip(g, 0.0, 1.0))
        t = int(round(GRIPPER_CLOSE_TICKS + frac * (GRIPPER_OPEN_TICKS - GRIPPER_CLOSE_TICKS)))
        if self.simulate:
            self._grip_sim = t
            return
        self._write(GRIPPER_ID, ADDR_GOAL_POSITION, 4, t)
        if wait:
            self._wait_servo(GRIPPER_ID, t, timeout)

    def set_roll(self, angle, wait=False, timeout=3.0):
        """Wrist roll (joint 5) to angle (rad), clipped to ROLL_LIM; torque must be on (ID 15). home() and torque(True) put
        it back to zero - the roll is masked unless you command it."""
        a = float(np.clip(angle, ROLL_LIM[0], ROLL_LIM[1]))
        t = int(round(ROLL_TICKS + a / RAD_PER_TICK))
        if self.simulate:
            self._roll_sim = a
            return
        self._write(ROLL_ID, ADDR_GOAL_POSITION, 4, t)
        if wait:
            self._wait_servo(ROLL_ID, t, timeout)

    def read_extra(self):
        """(roll angle in rad, gripper opening as a fraction 0 closed .. 1 open) - the two masked axes."""
        if self.simulate:
            return getattr(self, '_roll_sim', 0.0), (self._grip_sim - GRIPPER_CLOSE_TICKS) / (GRIPPER_OPEN_TICKS - GRIPPER_CLOSE_TICKS)
        roll = (_s32(self._read(ROLL_ID, ADDR_PRESENT_POSITION, 4)) - ROLL_TICKS) * RAD_PER_TICK
        g = _s32(self._read(GRIPPER_ID, ADDR_PRESENT_POSITION, 4))
        return roll, (g - GRIPPER_CLOSE_TICKS) / (GRIPPER_OPEN_TICKS - GRIPPER_CLOSE_TICKS)

    def _wait_servo(self, dxl_id, goal, timeout, tol=15, still_s=0.3):
        """Until one servo is within tol ticks of goal or has stopped moving (a gripper stops on an object)."""
        t0 = time.time(); prev = None; t_still = None
        while time.time() - t0 < timeout:
            time.sleep(0.03)
            p = _s32(self._read(dxl_id, ADDR_PRESENT_POSITION, 4))
            if abs(p - goal) <= tol:
                return
            if prev is not None and abs(p - prev) <= 1:
                t_still = t_still or time.time()
                if time.time() - t_still > still_s:
                    return
            else:
                t_still = None
            prev = p

    def home(self, wait=True):
        """Home pose (HOME_Q); from the rest region it unfolds via REST_VIA_Q first (see REST_Q)."""
        if not self.simulate:
            self._write(ROLL_ID, ADDR_GOAL_POSITION, 4, ROLL_TICKS)
        near_rest = np.max(np.abs(np.asarray(self.read_q()) - REST_Q)) < np.radians(25)
        if near_rest:
            self.move_q(REST_VIA_Q, wait=True)
        self.move_q(HOME_Q, wait=wait)

    def rest(self, wait=True):
        """Park in the folded rest pose (REST_Q), where the arm stays put with torque off. Use before torque(False).
        Route: home -> (close an open gripper there) -> REST_VIA_Q -> REST_Q, so the tip stays >= 22 mm above the table
        whatever order the joints finish in. Already at rest: nothing moves."""
        if np.max(np.abs(np.asarray(self.read_q()) - REST_Q)) < np.radians(5):
            return
        self.move_q(HOME_Q, wait=True)
        if self.read_extra()[1] > 0.2:
            self.set_gripper('closed', wait=True)
        self.move_q(REST_VIA_Q, wait=True)
        self.move_q(REST_Q, wait=wait)

    def close(self):
        if not self.simulate:
            self.port.closePort()
