# ELEC3901 Robotic Manipulators - lab notebooks

Your own copy of the lab notebooks. Commit your work and data files here every lab: later labs read them.

| Folder | What |
|---|---|
| `week00/` | The **pre-lab**: do it before week 1 (see its README). |
| `weekNN/` | This week's notebook. Work in it and commit it. |
| `data/` | Everything your notebooks save (measurements, calibration results, all tagged with the arm ID). Later labs read it. |
| `scripts/` | Helper modules, the mat files and `omx_ref.py`. **Do not edit**: staff replace this folder every week. |
| `environment.yml`, `requirements.txt`, `.vscode/` | The `elec3901` conda environment and the VS Code set-up. |

- **Before week 1:** the pre-lab in `week00/` (its README; Minerva "Pre-Lab Work" has screenshots): clone this repository in
  VS Code, install the recommended extensions when asked, create the `elec3901` conda environment, run `week00/check_setup.py`,
  run the notebook with the **elec3901** kernel, commit and push.
- **Start of every lab:** pull (Source Control > ... > Pull, or `git pull`) to receive the new week's folder. Open the notebook and
  run the cells at the top: **set-up** (finds `scripts/`, changes into `data/`), **connect** (the arm: it finds the USB port itself;
  set `ARM_ID` from the label on the base) and **camera** (the arm calibrates the camera from the mat - it moves, keep the workspace
  clear). Re-run the camera cell on its own to repeat the calibration; re-running connect is safe too.
- **End of every lab:** run the last cell (the arm parks and its torque goes off), then commit and **push** (Sync Changes).
- `scripts/omx_ref.py` holds reference versions of *previous* labs' tasks, so that last week never blocks this week.
- **Never commit** camera or calibration files you did not make, videos, or anything containing personal data.
- The assessment repository (issued in week 6) is separate. Copy across any code you want to reuse; it is not linked to this one.

Released so far: week 1.
