# 24_hyperalignment_surface_saving.py
# Loads alignment estimators from 23_hyperalignment_surface_run.py,
# transforms all subjects, and computes pairwise ISC before/after alignment.
# Analogous to 24_hyperalignment_smor_saving.py but for surface GIFTI data.

import gc
import os
import sys
import pickle
import joblib
import numpy as np
import nibabel as nib
from pathlib import Path
from itertools import combinations
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "prelim_discovery_analysis"))
from utils import TASKS, CONTRASTS, SUBJECTS, SESSIONS

# ── Paths ─────────────────────────────────────────────────────────────────────
SURFACE_BASE_DIR = Path('/scratch/groups/russpold/logben/discovery_bids/derivatives/lev1_surface')
ANALYSIS_DIR     = Path(__file__).parent
OUTPUT_DIR       = ANALYSIS_DIR / "hyperalignment_fmralign_results" / "discovery_sample" / "schaefer400_surface"
SAVE_DIR         = OUTPUT_DIR / "saved_arrays"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RT_CONTRAST        = "response_time"
N_VERTICES_PER_HEMI = 40962
train_subjects     = SUBJECTS[:-1]
left_out_subject   = SUBJECTS[-1]


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_surface_contrast_map_path(base_dir, subject, session, task, contrast, hemi, file_type="default"):
    file_ending = "z_score" if file_type == "z" else "effect-size"
    filename = (
        f'{subject}_{session}_task-{task}_run-1_hemi-{hemi}'
        f'_contrast-{contrast}_rtmodel-RTDur_stat-{file_ending}.func.gii'
    )
    return Path(base_dir) / subject / f'task-{task}' / 'indiv_contrasts' / filename


def vectorized_pearson(A, B):
    """Per-vertex Pearson r between two (n_contrasts, n_vertices) arrays."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    denom = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)) + 1e-10
    return (A * B).sum(0) / denom


def save_as_gifti(score, out_dir, stem):
    """Split (81924,) array into LH/RH and save as .func.gii files."""
    for hemi, data in [('L', score[:N_VERTICES_PER_HEMI]), ('R', score[N_VERTICES_PER_HEMI:])]:
        darray = nib.gifti.GiftiDataArray(data.astype(np.float32))
        img    = nib.gifti.GiftiImage(darrays=[darray])
        nib.save(img, out_dir / f'{stem}_hemi-{hemi}.func.gii')


# ─── 1. Load shared TCE index from run script ────────────────────────────────
# Load the exact tuple list used during alignment training so that subject
# arrays built here have the same row order and count as the fitted estimators.
print("Loading shared_tce_sorted from run script output...")
with open(OUTPUT_DIR / "shared_tce_sorted.pkl", "rb") as f:
    shared_tce_sorted = pickle.load(f)
print(f"Shared (task, contrast, encounter) tuples: {len(shared_tce_sorted)}")

# ─── 2. Load beta maps ────────────────────────────────────────────────────────
print("Loading beta maps...")
beta_maps = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for task in TASKS:
    for contrast in CONTRASTS[task]:
        for subject in SUBJECTS:
            enc_count = 0
            for session in SESSIONS:
                lh_path = build_surface_contrast_map_path(
                    SURFACE_BASE_DIR, subject, session, task, contrast, 'L'
                )
                rh_path = build_surface_contrast_map_path(
                    SURFACE_BASE_DIR, subject, session, task, contrast, 'R'
                )
                if lh_path.exists() and rh_path.exists():
                    enc_key = f"{enc_count + 1:02d}"
                    lh_data = nib.load(lh_path).darrays[0].data
                    rh_data = nib.load(rh_path).darrays[0].data
                    beta_maps[task][contrast][subject][enc_key] = np.concatenate([lh_data, rh_data])
                    enc_count += 1

# ─── 3. Build per-subject 2D arrays ──────────────────────────────────────────
print("Building subject arrays...")
subject_arrays = {}
for subj in SUBJECTS:
    rows = [beta_maps[task][contrast][subj][enc]
            for task, contrast, enc in shared_tce_sorted]
    subject_arrays[subj] = np.stack(rows, axis=0).astype(np.float32)
    print(f"  {subj}: {subject_arrays[subj].shape}")

del beta_maps
gc.collect()

# Save raw (unaligned) masked arrays
for subj, arr in subject_arrays.items():
    np.save(SAVE_DIR / f"masked_{subj}.npy", arr)
print(f"Saved unaligned arrays to {SAVE_DIR}")

# ─── 4. Load alignment estimators and transform ───────────────────────────────
print("Loading alignment estimators...")
template_estim = joblib.load(OUTPUT_DIR / "group_alignment.pkl")
pairwise_estim = joblib.load(OUTPUT_DIR / "pairwise_alignment.pkl")

print("Transforming subjects...")
masked_train     = {s: subject_arrays[s] for s in train_subjects}
aligned_train    = template_estim.transform(masked_train)
aligned_left_out = pairwise_estim.transform(subject_arrays[left_out_subject])
aligned_all      = {**aligned_train, left_out_subject: aligned_left_out}

for subj, arr in aligned_all.items():
    np.save(SAVE_DIR / f"aligned_{subj}.npy", arr)
print(f"Saved aligned arrays to {SAVE_DIR}")

# ─── 5. Pairwise ISC before and after alignment ───────────────────────────────
pairs = list(combinations(SUBJECTS, 2))
print(f"\nComputing ISC over {len(pairs)} pairs...")

isc_before_pairs = []
isc_after_pairs  = []
pair_labels      = []

for s1, s2 in pairs:
    r_before = vectorized_pearson(subject_arrays[s1], subject_arrays[s2])
    r_after  = vectorized_pearson(aligned_all[s1],    aligned_all[s2])
    isc_before_pairs.append(r_before)
    isc_after_pairs.append(r_after)
    pair_labels.append((s1, s2))
    print(f"  {s1} x {s2}: mean r before={r_before.mean():.3f}  after={r_after.mean():.3f}")

isc_before_mean = np.mean(isc_before_pairs, axis=0)
isc_after_mean  = np.mean(isc_after_pairs,  axis=0)
isc_diff        = isc_after_mean - isc_before_mean

print(f"\nOverall mean ISC before: {isc_before_mean.mean():.4f}")
print(f"Overall mean ISC after:  {isc_after_mean.mean():.4f}")
print(f"Mean ISC improvement:    {isc_diff.mean():.4f}")

# ─── 6. Save ──────────────────────────────────────────────────────────────────
np.save(SAVE_DIR / "isc_before_mean.npy",  isc_before_mean)
np.save(SAVE_DIR / "isc_after_mean.npy",   isc_after_mean)
np.save(SAVE_DIR / "isc_diff.npy",         isc_diff)
np.save(SAVE_DIR / "isc_before_pairs.npy", np.array(isc_before_pairs))
np.save(SAVE_DIR / "isc_after_pairs.npy",  np.array(isc_after_pairs))
with open(SAVE_DIR / "pair_labels.pkl", "wb") as f:
    pickle.dump(pair_labels, f)
print(f"Saved ISC arrays to {SAVE_DIR}")

save_as_gifti(isc_before_mean, SAVE_DIR, "isc_before_mean")
save_as_gifti(isc_after_mean,  SAVE_DIR, "isc_after_mean")
save_as_gifti(isc_diff,        SAVE_DIR, "isc_diff")
print(f"Saved ISC GIFTIs to {SAVE_DIR}")

print("\nDone.")
