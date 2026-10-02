# Pre-lab: get set up before week 1

Do this **before your first practical**, so the practical can be spent on the robot. It takes about 30 minutes, most of it
waiting for conda. The Minerva page "Pre-Lab Work" has the same steps with screenshots. It checks the whole workflow you use
every week: this repository, VS Code, the Python environment, Jupyter, commit and push.

## 1. Clone this repository and open it in VS Code

Accept the Classroom 50 invitation for **ELEC3901-LABS** (it creates your own private copy of this repository). In VS Code:
**Source Control** (left bar) > **Clone Repository** > paste your repository's URL (the green **Code** button on GitHub) >
choose a folder you can find again > **Open**.

When VS Code asks **"Do you want to install the recommended extensions for this repository?"**, choose **Install**
(Python, Jupyter; listed in `.vscode/extensions.json`).

## 2. Create the ELEC3901 environment (once per computer)

Lab PC: install **Anaconda** from AppsAnywhere, then open **Anaconda Prompt**. Your own laptop: install
[Miniconda](https://docs.anaconda.com/miniconda/) and open Anaconda Prompt (Windows) or Terminal (macOS, Linux).
Go to your repository's folder (`cd` followed by its path - the folder with `environment.yml` in it) and run:

```bash
conda env create -f environment.yml
conda activate elec3901
```

The first command takes several minutes. You only do it once: every lab uses the same `elec3901` environment. If a later
week changes `environment.yml`, update it with `conda env update -f environment.yml`.

## 3. Check your environment

With the environment active, in the repository folder:

```bash
python week00/check_setup.py
```

Every line should say `OK` (a `WARN` for mujoco is fine). You know it worked when you see:

```
############################################
    _[o_o]_    ELEC3901: all checks passed
   /|_____|\   commit setup_report.txt
############################################
```

If a line says `FAIL`, it tells you the fix. It writes `week00/setup_report.txt` (versions only, no personal data).

## 4. Run the notebook

Open `week00/getting_started.ipynb` in VS Code, choose the **elec3901** kernel (top right), run both cells, change
`THETA_DEG`, run them again and save.

## 5. Commit and push

In VS Code **Source Control**: press **+** next to `setup_report.txt` and `getting_started.ipynb` to stage them, type a
message that says what you did (for example `Environment checked; frame at 45 degrees`), press **Commit**, then
**Sync Changes**. Or in a terminal:

```bash
git add week00/setup_report.txt week00/getting_started.ipynb
git commit -m "Environment checked; frame at 45 degrees"
git push
```

Refresh your repository's page on GitHub: you should see your commit message next to both files.

## Every lab from now on

1. **Pull** (Source Control > ... > Pull, or `git pull`) at the start of every lab: each week's notebook arrives in a new
   `weekNN/` folder of this repository.
2. Work in VS Code with the elec3901 kernel; save as you go.
3. **Commit** when something works, and **push** (Sync Changes) before you leave the lab.
