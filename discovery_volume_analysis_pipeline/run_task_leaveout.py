"""
run_task_leaveout.py
====================
Leave-one-task-out hyperalignment on volumetric contrast maps.

For each of the 8 tasks (the "held-out" task):
  1. Fit iterative Procrustes GroupAlignment on ALL subjects using contrast maps
     from all OTHER tasks (all encounters, all non-RT contrasts).
  2. Apply the fitted per-subject rotations to the held-out task's contrast maps.
  3. Save the aligned (and unaligned) held-out arrays for downstream analysis.

Why this design:
  - The alignment never sees the held-out task's data, so ISC and practice-slope
    analyses on the aligned held-out data are properly cross-validated.
  - GroupAlignment learns one orthogonal rotation per subject from the training
    data; because rotations are linear operators, they generalise to any new data
    from the same subject without refitting.

Outputs (under results/task_leaveout/{task}/):
  masked_{subject}.npy   — unaligned held-out task arrays, (n_rows, n_voxels)
  aligned_{subject}.npy  — aligned held-out task arrays,   (n_rows, n_voxels)
  tce_order.pkl          — list of (task, contrast, enc) tuples defining row order
  training_tce_order.pkl — same for the training data (for reference)
"""

import gc
import sys
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from fmralign import GroupAlignment

# ── Config ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TASKS, CONTRASTS, SUBJECTS, RT_CONTRAST, RESULTS_DIR,
    setup_masker_and_labels, load_all_masked_data,
    find_shared_tce, build_subject_arrays,
)

LEAVEOUT_DIR = RESULTS_DIR / "task_leaveout"
LEAVEOUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load and mask all data ──────────────────────────────────────────────────
print("=" * 60)
print("Loading all contrast maps...")
print("=" * 60)

# Use any existing map as the reference for atlas resampling
from config import BASE_DIR, SESSIONS, build_first_level_contrast_map_path
reference_path = None
for task in TASKS:
    for contrast in CONTRASTS[task]:
        if contrast == RT_CONTRAST:
            continue
        for session in SESSIONS:
            p = build_first_level_contrast_map_path(
                BASE_DIR, SUBJECTS[0], session, task, contrast
            )
            if Path(p).exists():
                reference_path = p
                break
        if reference_path:
            break
    if reference_path:
        break

print(f"Reference image: {reference_path}")
reference_img = nib.load(reference_path)

# ── 2. Set up Schaefer 400 masker ──────────────────────────────────────────────
print("\nSetting up Schaefer 400 masker...")
masker, labels = setup_masker_and_labels(reference_img)

# ── 3. Load and mask every available contrast map ──────────────────────────────
print("\nLoading and masking all contrast maps...")
all_masked = load_all_masked_data(masker, verbose=True)
print(f"Total maps loaded: {len(all_masked)}")


# ── 4. Loop over held-out tasks ────────────────────────────────────────────────
for held_out_task in TASKS:
    print("\n" + "=" * 60)
    print(f"Held-out task: {held_out_task}")
    print("=" * 60)

    task_dir = LEAVEOUT_DIR / held_out_task
    task_dir.mkdir(exist_ok=True)

    # ── 4a. Find shared training tuples (all other tasks, all subjects) ──────
    # "Shared" means every subject has a map for that (task, contrast, encounter).
    other_tasks = [t for t in TASKS if t != held_out_task]
    training_tce = find_shared_tce(all_masked, SUBJECTS, tasks_subset=other_tasks)
    print(f"  Training tuples (shared across all subjects): {len(training_tce)}")

    # ── 4b. Build training arrays for all subjects ───────────────────────────
    training_arrays = build_subject_arrays(all_masked, SUBJECTS, training_tce)
    for subj, arr in training_arrays.items():
        print(f"    {subj}: training shape {arr.shape}")

    # ── 4c. Fit GroupAlignment (iterative Procrustes) on training data ───────
    # GroupAlignment starts from the Euclidean mean, then iteratively
    # aligns each subject to the group template and recomputes the mean.
    print("  Fitting GroupAlignment (iterative Procrustes)...")
    group_align = GroupAlignment(method="procrustes", labels=labels)
    group_align.fit(X=training_arrays, y="template")
    print("  Fit complete.")

    # ── 4d. Find shared held-out tuples (held-out task only, all subjects) ───
    leftout_tce = find_shared_tce(all_masked, SUBJECTS, tasks_subset=[held_out_task])
    print(f"  Held-out task tuples (shared across all subjects): {len(leftout_tce)}")

    if len(leftout_tce) == 0:
        print(f"  WARNING: no shared held-out tuples for {held_out_task}, skipping.")
        continue

    # ── 4e. Build held-out arrays ────────────────────────────────────────────
    leftout_arrays = build_subject_arrays(all_masked, SUBJECTS, leftout_tce)
    for subj, arr in leftout_arrays.items():
        print(f"    {subj}: held-out shape {arr.shape}")

    # ── 4f. Apply the fitted rotations to the held-out data ──────────────────
    # group_align stores one fitted estimator per subject (keyed by subject ID).
    # Calling transform() with new data applies each subject's rotation to that
    # subject's held-out arrays — the rotation generalises because it is a
    # linear, subject-specific operator learned from the other-task geometry.
    print("  Transforming held-out arrays...")
    aligned_leftout = group_align.transform(leftout_arrays)

    # ── 4g. Save ─────────────────────────────────────────────────────────────
    for subj in SUBJECTS:
        np.save(task_dir / f"masked_{subj}.npy",  leftout_arrays[subj])
        np.save(task_dir / f"aligned_{subj}.npy", aligned_leftout[subj])

    with open(task_dir / "tce_order.pkl", "wb") as f:
        pickle.dump(leftout_tce, f)
    with open(task_dir / "training_tce_order.pkl", "wb") as f:
        pickle.dump(training_tce, f)

    print(f"  Saved to {task_dir}")

    # Free the fitted estimator before the next iteration
    del group_align, training_arrays, leftout_arrays, aligned_leftout
    gc.collect()

print("\n" + "=" * 60)
print("Task leave-out alignment complete.")
print("=" * 60)
