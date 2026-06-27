"""
run_encounter_leaveout.py
=========================
Leave-one-encounter-out hyperalignment on volumetric contrast maps.

For each encounter N in [01, 02, 03, 04, 05]:
  1. Fit iterative Procrustes GroupAlignment on ALL subjects using contrast maps
     from ALL tasks, ALL encounters EXCEPT encounter N.
  2. Apply the fitted per-subject rotations to encounter N's maps (all tasks).
  3. Save the aligned (and unaligned) encounter-N arrays for downstream analysis.

Why this design:
  - Each encounter's data is aligned using a model that never saw that encounter,
    giving a properly cross-validated basis for ISC and RSA across encounters.
  - Because the rotation is a fixed linear operator per subject, it generalises
    from other-encounter data to the held-out encounter.

Outputs (under results/encounter_leaveout/enc{N}/):
  masked_{subject}.npy   — unaligned encounter-N arrays, (n_shared_tc, n_voxels)
  aligned_{subject}.npy  — aligned encounter-N arrays,   (n_shared_tc, n_voxels)
  tc_order.pkl           — list of (task, contrast) tuples defining row order
  training_tce_order.pkl — (task, contrast, enc) tuples used to fit alignment
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
    TASKS, CONTRASTS, SUBJECTS, ENCOUNTERS, RT_CONTRAST, RESULTS_DIR,
    setup_masker_and_labels, load_all_masked_data,
    find_shared_tce, build_subject_arrays,
)

LEAVEOUT_DIR = RESULTS_DIR / "encounter_leaveout"
LEAVEOUT_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load reference image and set up masker ──────────────────────────────────
print("=" * 60)
print("Setting up masker...")
print("=" * 60)

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
reference_img  = nib.load(reference_path)
masker, labels = setup_masker_and_labels(reference_img)

# ── 2. Load and mask all contrast maps ────────────────────────────────────────
print("\nLoading and masking all contrast maps...")
all_masked = load_all_masked_data(masker, verbose=True)
print(f"Total maps loaded: {len(all_masked)}")


# ── 3. Loop over held-out encounters ──────────────────────────────────────────
for held_out_enc in ENCOUNTERS:
    print("\n" + "=" * 60)
    print(f"Held-out encounter: {held_out_enc}")
    print("=" * 60)

    enc_dir = LEAVEOUT_DIR / f"enc{held_out_enc}"
    enc_dir.mkdir(exist_ok=True)

    # ── 3a. Find shared training tuples (all tasks, all OTHER encounters) ────
    other_encs   = [e for e in ENCOUNTERS if e != held_out_enc]
    training_tce = find_shared_tce(
        all_masked, SUBJECTS,
        tasks_subset=None,         # all tasks
        encounters_subset=other_encs,
    )
    print(f"  Training tuples (shared, excluding enc {held_out_enc}): {len(training_tce)}")

    # ── 3b. Build training arrays ─────────────────────────────────────────────
    training_arrays = build_subject_arrays(all_masked, SUBJECTS, training_tce)
    for subj, arr in training_arrays.items():
        print(f"    {subj}: training shape {arr.shape}")

    # ── 3c. Fit GroupAlignment ────────────────────────────────────────────────
    print("  Fitting GroupAlignment (iterative Procrustes)...")
    group_align = GroupAlignment(method="procrustes", labels=labels)
    group_align.fit(X=training_arrays, y="template")
    print("  Fit complete.")

    # ── 3d. Find shared (task, contrast) pairs at held-out encounter ──────────
    # Only include pairs present for ALL subjects at this encounter.
    task_order   = {t: i for i, t in enumerate(TASKS)}
    available_tc = {subj: set() for subj in SUBJECTS}
    for (subj, task, contrast, enc), _ in all_masked.items():
        if enc == held_out_enc and subj in SUBJECTS:
            available_tc[subj].add((task, contrast))

    shared_tc = set.intersection(*available_tc.values())
    tc_order  = sorted(
        shared_tc,
        key=lambda tc: (task_order[tc[0]], CONTRASTS[tc[0]].index(tc[1])),
    )
    print(f"  Shared (task, contrast) pairs at enc {held_out_enc}: {len(tc_order)}")

    if len(tc_order) == 0:
        print(f"  WARNING: no shared pairs at encounter {held_out_enc}, skipping.")
        continue

    # ── 3e. Build held-out encounter arrays ───────────────────────────────────
    leftout_arrays = {
        subj: np.stack([
            all_masked[(subj, task, contrast, held_out_enc)]
            for task, contrast in tc_order
        ])
        for subj in SUBJECTS
    }
    for subj, arr in leftout_arrays.items():
        print(f"    {subj}: held-out shape {arr.shape}")

    # ── 3f. Apply fitted rotations to held-out encounter ──────────────────────
    print("  Transforming held-out encounter arrays...")
    aligned_leftout = group_align.transform(leftout_arrays)

    # ── 3g. Save ──────────────────────────────────────────────────────────────
    for subj in SUBJECTS:
        np.save(enc_dir / f"masked_{subj}.npy",  leftout_arrays[subj])
        np.save(enc_dir / f"aligned_{subj}.npy", aligned_leftout[subj])

    with open(enc_dir / "tc_order.pkl", "wb") as f:
        pickle.dump(tc_order, f)
    with open(enc_dir / "training_tce_order.pkl", "wb") as f:
        pickle.dump(training_tce, f)

    print(f"  Saved to {enc_dir}")

    del group_align, training_arrays, leftout_arrays, aligned_leftout
    gc.collect()

print("\n" + "=" * 60)
print("Encounter leave-out alignment complete.")
print("=" * 60)
