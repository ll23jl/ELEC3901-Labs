#!/usr/bin/env python
"""
ELEC3901 environment check. Run it from this folder, in the environment you will use for the labs:

    python check_setup.py

It checks Python, the packages the labs need and your git identity, prints what it found, and writes setup_report.txt
(versions only, no personal data) for you to commit. Nothing is installed or changed.
"""
import importlib
import importlib.metadata as md
import platform
import subprocess
import sys

PACKAGES = [('numpy', 'numpy', True), ('scipy', 'scipy', True), ('matplotlib', 'matplotlib', True), ('cv2', 'opencv-contrib-python', True), ('spatialmath', 'spatialmath-python', True), ('roboticstoolbox', 'roboticstoolbox-python', True), ('swift', 'swift-sim', True), ('dynamixel_sdk', 'dynamixel-sdk', True), ('serial', 'pyserial', True), ('pygrabber', 'pygrabber', 'win32'), ('jupyterlab', 'jupyterlab', True), ('mujoco', 'mujoco', False)]

ok, lines = True, []


def report(status, what, detail=''):
    global ok
    if status == 'FAIL':
        ok = False
    lines.append(f'{status:5} {what:28} {detail}')
    print(lines[-1])


print('ELEC3901 environment check\n')
v = sys.version_info
print(f'Environment: {sys.executable}\n')   # on screen only: the path can contain your username
report('OK' if v >= (3, 10) else 'FAIL', 'Python', platform.python_version() + ('' if v >= (3, 10) else ' - need 3.10 or newer'))
for name, pip_name, required in PACKAGES:
    if required == 'win32':
        if sys.platform != 'win32':
            continue                                     # Windows-only package
        required = True
    try:
        m = importlib.import_module(name)
        try:
            ver = md.version(pip_name)
        except md.PackageNotFoundError:
            ver = getattr(m, '__version__', '?')
        problem = ('no ArUco: pip uninstall opencv-python, then pip install opencv-contrib-python'
                   if name == 'cv2' and not hasattr(m, 'aruco') else '')
        report('FAIL' if problem else 'OK', pip_name, problem or ver)
    except Exception as e:  # ImportError, or a broken install that fails on import
        report('FAIL' if required else 'WARN', pip_name,
               f'not importable ({e.__class__.__name__}): pip install {pip_name}'
               + ('' if required else '  (optional: the MuJoCo twin for home practice)'))


def git(*args):
    try:
        return subprocess.run(['git', *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


gv = git('--version')
report('OK' if gv else 'FAIL', 'git', gv or 'not found: install git (https://git-scm.com) and reopen the terminal')
if gv:
    name, email = git('config', 'user.name'), git('config', 'user.email')
    report('OK' if name else 'FAIL', 'git user.name', 'set' if name else 'not set: git config --global user.name "Your Name"')
    report('OK' if email else 'FAIL', 'git user.email',
           ('set' + ('' if email.endswith('leeds.ac.uk') else ' (use the email on your GitHub account)')) if email
           else 'not set: git config --global user.email "you@leeds.ac.uk"')

import os
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'setup_report.txt'), 'w', encoding='utf-8') as f:   # beside this script
    f.write(f'{platform.system()} {platform.release()} {platform.machine()}\n' + '\n'.join(lines) + '\n'
            + ('ALL CHECKS PASSED\n' if ok else 'SOME CHECKS FAILED\n'))

print()
if ok:
    print('############################################')
    print('    _[o_o]_    ELEC3901: all checks passed ')
    print('   /|_____|\\   commit setup_report.txt     ')
    print('############################################')
else:
    print('Some checks FAILED (see above). Fix them and run this again, or ask in the practical.')
sys.exit(0 if ok else 1)
