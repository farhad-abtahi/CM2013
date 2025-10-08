# Setup Guide: Python, Conda, and GitHub Desktop

This guide will help you set up your development environment for the Sleep Scoring Project. You'll learn how to install Python, manage environments with Conda, and use GitHub Desktop for version control.

---

## Table of Contents
1. [Installing Python and Anaconda](#installing-python-and-anaconda)
2. [Managing Environments with Conda](#managing-environments-with-conda)
3. [Installing Project Dependencies](#installing-project-dependencies)
4. [Using GitHub Desktop](#using-github-desktop)
5. [Troubleshooting](#troubleshooting)

---

## Installing Python and Anaconda

### What is Anaconda?

Anaconda is a distribution of Python that comes with:
- Python interpreter
- Conda package manager
- 250+ pre-installed scientific packages
- Environment management tools
- Jupyter Notebook

**Why use Anaconda for this project?**
- Isolate project dependencies (avoid conflicts with other Python projects)
- Easy to install scientific packages (NumPy, SciPy, scikit-learn)
- Cross-platform (works on Windows, macOS, Linux)
- Manage multiple Python versions

### Installation Steps

#### Windows

1. **Download Anaconda:**
   - Visit: https://www.anaconda.com/download
   - Download the Windows installer (64-bit recommended)
   - File size: ~500 MB

2. **Run the installer:**
   - Double-click the downloaded `.exe` file
   - Click "Next" through the welcome screens
   - **Important:** Check "Add Anaconda to my PATH environment variable" (makes conda accessible from command prompt)
   - Choose installation location (default is fine: `C:\Users\YourName\anaconda3`)
   - Click "Install" (takes 5-10 minutes)

3. **Verify installation:**
   - Open "Anaconda Prompt" from Start Menu
   - Type: `conda --version`
   - You should see: `conda 23.x.x` or similar

#### macOS

1. **Download Anaconda:**
   - Visit: https://www.anaconda.com/download
   - Download the macOS installer (Intel or Apple Silicon)
   - File size: ~500 MB

2. **Run the installer:**
   - Double-click the downloaded `.pkg` file
   - Follow installation wizard
   - Choose "Install for me only" (recommended)
   - Installation location: `/Users/YourName/anaconda3`

3. **Verify installation:**
   - Open Terminal (Applications → Utilities → Terminal)
   - Type: `conda --version`
   - You should see: `conda 23.x.x` or similar
   - If command not found, run: `source ~/anaconda3/bin/activate`

#### Linux (Ubuntu/Debian)

1. **Download Anaconda:**
   - Visit: https://www.anaconda.com/download
   - Download the Linux installer (64-bit)
   - File size: ~500 MB

2. **Install via terminal:**
   ```bash
   cd ~/Downloads
   bash Anaconda3-2024.xx-Linux-x86_64.sh
   ```
   - Press Enter to review license
   - Type "yes" to accept
   - Press Enter to confirm installation location
   - Type "yes" to initialize conda

3. **Verify installation:**
   - Close and reopen terminal
   - Type: `conda --version`
   - You should see: `conda 23.x.x` or similar

---

## Managing Environments with Conda

### What is a Conda Environment?

A conda environment is an isolated workspace containing:
- Specific Python version
- Specific package versions
- Independent from other projects

**Why use environments?**
- Avoid package conflicts between projects
- Reproducible setup (same versions as teammates)
- Easy to reset if something breaks
- Test different package versions

### Creating Your Project Environment

#### Step 1: Create a New Environment

Open Anaconda Prompt (Windows) or Terminal (macOS/Linux) and run:

```bash
conda create -n sleep-scoring python=3.10
```

**What this does:**
- `conda create`: Creates a new environment
- `-n sleep-scoring`: Names the environment "sleep-scoring"
- `python=3.10`: Installs Python version 3.10

**Expected output:**
```
Collecting package metadata...
Solving environment...
Proceed ([y]/n)?
```
Type `y` and press Enter.

Installation takes 2-3 minutes.

#### Step 2: Activate the Environment

**Windows (Anaconda Prompt):**
```bash
conda activate sleep-scoring
```

**macOS/Linux (Terminal):**
```bash
conda activate sleep-scoring
```

**You'll know it worked when you see:**
```
(sleep-scoring) C:\Users\YourName>
```
or
```
(sleep-scoring) username@computer:~$
```

The `(sleep-scoring)` prefix shows your active environment.

#### Step 3: Verify Python Version

```bash
python --version
```

Should output: `Python 3.10.x`

### Common Conda Commands

#### List All Environments
```bash
conda env list
```

Output shows all environments (active one has `*`):
```
# conda environments:
#
base                     /Users/YourName/anaconda3
sleep-scoring         *  /Users/YourName/anaconda3/envs/sleep-scoring
```

#### Deactivate Current Environment
```bash
conda deactivate
```

Returns you to the `(base)` environment.

#### Delete an Environment
```bash
conda env remove -n sleep-scoring
```

Use this if you need to start fresh.

#### Export Environment (Share with Team)
```bash
conda env export > environment.yml
```

Creates a file with all installed packages and versions.

#### Create Environment from File
```bash
conda env create -f environment.yml
```

Recreates the exact same environment on another computer.

---

## Installing Project Dependencies

### Method 1: Using requirements.txt (Recommended)

1. **Activate your environment:**
   ```bash
   conda activate sleep-scoring
   ```

2. **Navigate to project directory:**
   ```bash
   cd /path/to/CM2013/Python
   ```

3. **Install all dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

**What's in requirements.txt?**
```
numpy
scipy
scikit-learn
matplotlib
joblib
mne
```

These packages will be installed in your `sleep-scoring` environment only.

### Method 2: Installing Packages Individually

If you need to install packages one at a time:

#### Using conda (preferred for scientific packages):
```bash
conda install numpy scipy scikit-learn matplotlib
```

#### Using pip:
```bash
pip install mne joblib
```

**When to use conda vs pip?**
- **conda**: Large scientific packages (NumPy, SciPy, scikit-learn) - optimized binaries
- **pip**: Specialized packages not in conda (mne, some ML libraries)

### Verifying Installation

Test that packages are installed correctly:

```bash
python -c "import numpy; print(numpy.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import mne; print(mne.__version__)"
```

Each should print a version number without errors.

### Running the Project Tests

Verify your setup by running the project tests:

```bash
cd Python
python -m pytest tests/ -v
```

**Expected output:**
```
tests/test_config.py::test_config_loads PASSED
tests/test_data_loader.py::test_dummy_data_loads PASSED
...
======================== 13 passed in 2.5s ========================
```

If all tests pass, your environment is correctly configured!

---

## Using GitHub Desktop

### What is GitHub Desktop?

GitHub Desktop is a graphical application for version control using Git. It makes collaboration easier without command-line complexity.

**Why use GitHub Desktop for this project?**
- Visual interface for Git operations
- Easy to see file changes
- Simple branch management
- No need to memorize Git commands
- Integrated with GitHub.com

### Installation

#### Windows & macOS

1. **Download GitHub Desktop:**
   - Visit: https://desktop.github.com/
   - Click "Download for [Your OS]"
   - File size: ~100-150 MB

2. **Install:**
   - **Windows:** Run the `.exe` installer
   - **macOS:** Drag the app to Applications folder

3. **Sign in:**
   - Open GitHub Desktop
   - Click "Sign in to GitHub.com"
   - Enter your GitHub credentials
   - Authorize GitHub Desktop

### Getting Started with Your Project

#### Option 1: Clone the Repository (If it already exists)

1. **Clone from GitHub:**
   - Open GitHub Desktop
   - Click "File" → "Clone repository"
   - Select "URL" tab
   - Enter repository URL: `https://github.com/username/CM2013`
   - Choose local path (e.g., `C:\Users\YourName\Documents\CM2013`)
   - Click "Clone"

2. **Verify:**
   - Repository appears in left sidebar
   - Files are downloaded to your local path

#### Option 2: Create a New Repository

1. **Create locally:**
   - Open GitHub Desktop
   - Click "File" → "New repository"
   - Name: `CM2013`
   - Local path: Choose where to save
   - Initialize with README: ✓ (checked)
   - Git ignore: Python
   - Click "Create repository"

2. **Add your files:**
   - Copy your project files to the repository folder
   - GitHub Desktop automatically detects changes

3. **Publish to GitHub:**
   - Click "Publish repository" (top right)
   - Choose: Public or Private
   - Add description (optional)
   - Click "Publish repository"

### Daily Workflow with GitHub Desktop

#### 1. Check for Updates (Pull)

**Before you start working each day:**

- Open GitHub Desktop
- Select your repository
- Click "Fetch origin" (top right)
- If changes available, click "Pull origin"

**Why?** Gets latest changes from teammates.

#### 2. Make Changes

- Edit your code in your favorite editor (VS Code, PyCharm, etc.)
- Save files as usual

#### 3. Review Changes

- Open GitHub Desktop
- Left panel shows "Changes" tab
- Click on each file to see what changed:
  - Green lines: Added
  - Red lines: Deleted
  - Modified lines: Red (old) + Green (new)

#### 4. Commit Changes

**What is a commit?** A snapshot of your work with a descriptive message.

1. **Write commit message:**
   - Bottom left: "Summary" field
   - Example: `"Implement bandpass filter for EEG preprocessing"`
   - Optional: Add longer "Description"

2. **Commit:**
   - Click "Commit to main" button
   - Changes are saved locally

**Commit message best practices:**
- Start with a verb (Add, Fix, Update, Remove)
- Be specific: "Fix baseline wander filter cutoff" not "Fixed stuff"
- Keep summary under 50 characters

#### 5. Push to GitHub

**After committing:**

- Click "Push origin" (top right)
- Uploads your commits to GitHub.com
- Now teammates can see your changes

**When to push?**
- After completing a feature
- End of work session
- Before switching computers

### Working with Branches

**What is a branch?** A parallel version of your code for developing features.

#### Create a New Branch

1. Click "Current branch" dropdown (top center)
2. Click "New branch"
3. Name: `feature/preprocessing` or `fix/notch-filter`
4. Click "Create branch"

#### Switch Between Branches

1. Click "Current branch" dropdown
2. Select branch to switch to
3. Your files update automatically

#### Merge Branches

**After finishing a feature:**

1. Switch to `main` branch
2. Click "Branch" menu → "Merge into current branch"
3. Select feature branch
4. Click "Create merge commit"
5. Push to GitHub

### Handling Merge Conflicts

**What is a merge conflict?** Two people edited the same lines.

**GitHub Desktop will show:**
- ⚠️ Warning icon on files with conflicts
- "Resolve conflicts" button

**To resolve:**

1. Click "Open in [Editor]"
2. Look for conflict markers:
   ```python
   <<<<<<< HEAD
   Your changes
   =======
   Teammate's changes
   >>>>>>> branch-name
   ```
3. Choose which version to keep (or combine both)
4. Delete conflict markers
5. Save file
6. Return to GitHub Desktop → Click "Continue merge"

### Team Collaboration Tips

#### Setting Up Team Repository

1. **One person creates repository:**
   - Follow "Create a New Repository" steps above
   - Click "Publish repository" → Choose "Private"

2. **Add team members:**
   - Go to GitHub.com
   - Navigate to repository
   - Click "Settings" → "Collaborators"
   - Click "Add people"
   - Enter teammate GitHub usernames

3. **Team members clone:**
   - Each teammate opens GitHub Desktop
   - Click "File" → "Clone repository"
   - Select the team repository
   - Choose local path

#### Branch Strategy for Teams

**Main branch:**
- Always working code
- No direct commits (use branches instead)

**Feature branches:**
- One branch per person or feature
- Name format: `feature/description` or `yourname/description`
- Example: `john/preprocessing`, `feature/svm-classifier`

**Workflow:**
1. Create feature branch
2. Work on your feature
3. Commit and push regularly
4. When done, merge to main
5. Delete feature branch

#### Staying Synchronized

**Pull frequently:**
- Start of each work session: Fetch + Pull
- Before pushing: Fetch + Pull (avoid conflicts)
- After teammate notifies of push: Pull immediately

**Communicate:**
- Notify team in chat when pushing major changes
- Use commit messages to explain what you did
- Review each other's changes before merging to main

### GitHub Desktop Interface Guide

**Top Bar:**
- **Current repository:** Dropdown to switch between projects
- **Current branch:** Shows active branch, click to switch
- **Fetch origin:** Check for remote changes
- **Pull origin:** Download remote changes (appears after Fetch if updates available)
- **Push origin:** Upload local commits (appears after local commits)

**Left Sidebar:**
- **Changes tab:** Files you've modified (not yet committed)
- **History tab:** Past commits (click to view details)

**Main Area:**
- **File list:** Shows changed files with icons (M=Modified, A=Added, D=Deleted)
- **Diff viewer:** Shows line-by-line changes when you click a file

**Bottom Left:**
- **Commit message fields:** Summary and description
- **Commit button:** Saves current changes

---

## Troubleshooting

### Conda Issues

#### "conda: command not found" (macOS/Linux)

**Solution:**
```bash
export PATH="$HOME/anaconda3/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc on newer macOS
```

Or add to your shell profile:
```bash
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

#### "conda: command not found" (Windows)

**Solution:**
1. Search for "Environment Variables" in Start Menu
2. Click "Edit the system environment variables"
3. Click "Environment Variables" button
4. Under "User variables", find "Path"
5. Click "Edit" → "New"
6. Add: `C:\Users\YourName\anaconda3\Scripts`
7. Click OK on all windows
8. Restart Anaconda Prompt

#### Environment activation not working

**Solution:**
```bash
conda init
```

Close and reopen your terminal, then try again.

#### Package installation fails

**Try these steps:**

1. **Update conda:**
   ```bash
   conda update conda
   ```

2. **Install from conda-forge:**
   ```bash
   conda install -c conda-forge package-name
   ```

3. **Use pip as fallback:**
   ```bash
   pip install package-name
   ```

### Python/Package Issues

#### ImportError: No module named 'package'

**Causes:**
- Package not installed
- Wrong environment activated
- Package installed in different environment

**Solution:**
```bash
# Verify active environment
conda env list

# Activate correct environment
conda activate sleep-scoring

# Install missing package
pip install package-name

# Verify installation
pip list | grep package-name
```

#### Multiple Python installations conflict

**Solution:** Use conda environments exclusively:
```bash
# Don't use system Python
# Always activate environment first
conda activate sleep-scoring
which python  # Should show path inside anaconda3/envs/sleep-scoring
```

#### Tests fail with "ModuleNotFoundError"

**Solution:**
```bash
# Make sure you're in the correct directory
cd CM2013/Python

# Install pytest if needed
pip install pytest

# Run tests with verbose output
python -m pytest tests/ -v
```

### GitHub Desktop Issues

#### "Authentication failed"

**Solution:**
1. Click "File" → "Options" (Windows) or "Preferences" (macOS)
2. Click "Accounts"
3. Click "Sign out"
4. Sign in again
5. Re-authorize GitHub Desktop in browser

#### "Push rejected" error

**Cause:** Remote has changes you don't have locally.

**Solution:**
1. Click "Fetch origin"
2. Click "Pull origin"
3. Resolve any conflicts
4. Try push again

#### Can't see changes in GitHub Desktop

**Solution:**
1. Verify you're viewing correct repository (top left dropdown)
2. Check you're on correct branch (top center dropdown)
3. Verify files are saved in correct folder
4. Click "Repository" → "Refresh" (or press F5)

#### Merge conflict seems stuck

**Solution:**
1. Open file in text editor
2. Search for `<<<<<<<` markers
3. Ensure all conflict markers are removed
4. Save file
5. Return to GitHub Desktop
6. File should disappear from conflicts list

#### Branch doesn't appear

**Solution:**
```bash
# Fetch all branches
Click "Fetch origin"

# If still missing, it may be only local
# Check in History tab → All branches
```

### General Tips

#### Starting Fresh

If everything is broken, reset your environment:

```bash
# Deactivate current environment
conda deactivate

# Delete environment
conda env remove -n sleep-scoring

# Recreate environment
conda create -n sleep-scoring python=3.10

# Activate
conda activate sleep-scoring

# Reinstall packages
cd CM2013/Python
pip install -r requirements.txt
```

#### Verify Setup Checklist

Use this checklist to confirm everything works:

- [ ] Conda installed: `conda --version`
- [ ] Environment created: `conda env list` shows `sleep-scoring`
- [ ] Environment activated: Prompt shows `(sleep-scoring)`
- [ ] Python correct version: `python --version` shows 3.10.x
- [ ] Packages installed: `pip list` shows numpy, scipy, sklearn, mne
- [ ] Tests pass: `python -m pytest tests/ -v` shows all passed
- [ ] GitHub Desktop installed and signed in
- [ ] Repository cloned to local computer
- [ ] Can commit and push changes

---

## Additional Resources

### Conda Documentation
- Official docs: https://docs.conda.io/
- Cheat sheet: https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html
- Managing environments: https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

### GitHub Desktop Documentation
- Official docs: https://docs.github.com/en/desktop
- Getting started: https://docs.github.com/en/desktop/installing-and-configuring-github-desktop
- Video tutorials: https://www.youtube.com/githubguides

### Python Package Documentation
- NumPy: https://numpy.org/doc/
- SciPy: https://docs.scipy.org/doc/scipy/
- scikit-learn: https://scikit-learn.org/stable/
- MNE-Python: https://mne.tools/stable/index.html

### Git and Version Control
- Git basics: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
- GitHub guides: https://guides.github.com/
- Interactive Git tutorial: https://learngitbranching.js.org/

---

## Getting Help

**For setup issues:**
1. Check this troubleshooting section first
2. Search error messages on Google/Stack Overflow
3. Ask teammates if they encountered similar issues
4. Post in course forum with error message and what you tried

**For project-specific questions:**
- Refer to PROJECT_GUIDE.md for implementation guidance
- Check Python/README.md or MATLAB/README.md for language-specific help

**Good luck with your setup! 🚀**
