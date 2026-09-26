# Safe Git Migration

Your existing `main` branch contains earlier V3/V4 work. The safest approach is to publish this cleaned final project as a **new branch first**, verify it in Kaggle, and only then decide whether to make it your default branch.

## Recommended Windows workflow

Assume the clean ZIP has been extracted to:

```text
C:\Prac_Python\World_Models_Final64_Clean
```

Create a disposable fresh clone of your existing GitHub repository:

```cmd
cd C:\Prac_Python
git clone https://github.com/GothamSaiyanT/world_models.git world_models_final64_git
cd world_models_final64_git
```

Create a clean orphan branch so earlier project files are not part of the final branch tree:

```cmd
git switch --orphan final64-clean
git rm -rf --ignore-unmatch .
```

Copy the clean project into this Git working directory. In Windows CMD:

```cmd
robocopy C:\Prac_Python\World_Models_Final64_Clean C:\Prac_Python\world_models_final64_git /E /XD .git
```

`robocopy` commonly returns exit code 1 when files were copied successfully; that is not an error.

Now verify that no data/model binaries are staged:

```cmd
git add .
git status --short
git ls-files | findstr /I "frames.npy .npz .pt .mp4"
```

The final `findstr` command should print nothing.

Commit and push:

```cmd
git commit -m "Final clean 64x64 world model experiment"
git push -u origin final64-clean
```

## Kaggle clone

Use a shallow clone so Kaggle downloads only the clean branch snapshot:

```bash
!git clone --depth 1 -b final64-clean https://github.com/GothamSaiyanT/world_models.git
```

This avoids downloading the earlier project history during the experiment.

## After verification

Keep `main` as a backup while you finish the paper and video. There is no need to force-push or delete the old history before submission.
