# 24_hyperalignment_smor_saving.py
# Same as 21_hyperalignment_saving.py but uses the smorgasbord atlas mask
# and loads alignment estimators from smorgasbord_parcels/.

import os
import sys
import gc
import pickle
import joblib
import numpy as np
import nibabel as nib
from pathlib import Path
from itertools import combinations
from collections import defaultdict
from nilearn.maskers import NiftiMasker
from nilearn.image import concat_imgs, math_img, resample_to_img
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    TASKS, CONTRASTS, SUBJECTS, SESSIONS,
    build_first_level_contrast_map_path,
)
from config import BASE_DIR

ANALYSIS_DIR     = Path(__file__).parent
SMOR_ATLAS_NII   = ANALYSIS_DIR / "smorgasbord_atlas_files" / "tpl-MNI152NLin2009cAsym_res-01_atlas-smorgasbord_dseg.nii"
OUTPUT_DIR       = ANALYSIS_DIR / "hyperalignment_fmralign_results" / "discovery_sample"
ALIGN_DIR        = OUTPUT_DIR / "smorgasbord_parcels"
SAVE_DIR         = ALIGN_DIR / "saved_arrays"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RT_CONTRAST      = "response_time"
train_subjects   = SUBJECTS[:-1]
left_out_subject = SUBJECTS[-1]

# ─── 1. Load beta maps ────────────────────────────────────────────────────────
print("Loading beta maps...")
beta_maps = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
for task in TASKS:
    for contrast in CONTRASTS[task]:
        for subject in SUBJECTS:
            enc_count = 0
            for session in SESSIONS:
                path = build_first_level_contrast_map_path(BASE_DIR, subject, session, task, contrast)
                if os.path.exists(path):
                    enc_key = f"{enc_count + 1:02d}"
                    beta_maps[task][contrast][subject][enc_key] = nib.load(path)
                    enc_count += 1

# ─── 2. Find shared (task, contrast, encounter) tuples ───────────────────────
task_order = {t: i for i, t in enumerate(TASKS)}
available = {subj: set() for subj in SUBJECTS}
for subj in SUBJECTS:
    for task in TASKS:
        for contrast in CONTRASTS[task]:
            if contrast == RT_CONTRAST:
                continue
            for enc_key in beta_maps[task][contrast].get(subj, {}):
                available[subj].add((task, contrast, enc_key))

shared_tce = set.intersection(*available.values())
shared_tce_sorted = sorted(
    shared_tce,
    key=lambda tce: (task_order[tce[0]], CONTRASTS[tce[0]].index(tce[1]), tce[2])
)
print(f"Shared tuples: {len(shared_tce_sorted)}")

with open(SAVE_DIR / "shared_tce_sorted.pkl", "wb") as f:
    pickle.dump(shared_tce_sorted, f)

# ─── 3. Set up masker from smorgasbord atlas footprint ───────────────────────
print("Setting up masker...")
reference_img = beta_maps[TASKS[0]][CONTRASTS[TASKS[0]][0]][SUBJECTS[0]]["01"]
atlas_resampled = resample_to_img(
    nib.load(SMOR_ATLAS_NII), reference_img,
    interpolation="nearest", force_resample=True, copy_header=True,
)

mask_img = math_img("atlas > 0", atlas=atlas_resampled)
masker = NiftiMasker(mask_img=mask_img).fit()
print(f"Mask covers {masker.mask_img_.get_fdata().sum():.0f} voxels")

# ─── 4. Build masked arrays for all subjects ─────────────────────────────────
print("Masking subjects...")
masked_all = {}
for subj in SUBJECTS:
    imgs = [beta_maps[task][contrast][subj][enc] for task, contrast, enc in shared_tce_sorted]
    img_4d = concat_imgs(imgs, dtype=np.float32)
    masked_all[subj] = masker.transform(img_4d).astype(np.float32)
    del img_4d
    gc.collect()
    print(f"  {subj}: {masked_all[subj].shape}")

del beta_maps
gc.collect()

for subj, arr in masked_all.items():
    np.save(SAVE_DIR / f"masked_{subj}.npy", arr)
print(f"Saved masked arrays to {SAVE_DIR}")

# ─── 5. Load alignment estimators and transform ──────────────────────────────
print("Loading alignment estimators...")
template_estim  = joblib.load(ALIGN_DIR / "group_alignment.pkl")
pairwise_estim  = joblib.load(ALIGN_DIR / "pairwise_alignment.pkl")

print("Transforming subjects...")
masked_train     = {s: masked_all[s] for s in train_subjects}
aligned_train    = template_estim.transform(masked_train)
aligned_left_out = pairwise_estim.transform(masked_all[left_out_subject])
aligned_all      = {**aligned_train, left_out_subject: aligned_left_out}

for subj, arr in aligned_all.items():
    np.save(SAVE_DIR / f"aligned_{subj}.npy", arr)
print(f"Saved aligned arrays to {SAVE_DIR}")

# ─── 6. Compute pairwise ISC before and after ────────────────────────────────
def vectorized_pearson(A, B):
    """Per-voxel Pearson r between two (n_contrasts x n_voxels) arrays."""
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    denom = np.sqrt((A**2).sum(0) * (B**2).sum(0)) + 1e-10
    return (A * B).sum(0) / denom  # (n_voxels,)

pairs = list(combinations(SUBJECTS, 2))
print(f"\nComputing ISC over {len(pairs)} pairs...")

isc_before_pairs = []
isc_after_pairs  = []
pair_labels      = []

for s1, s2 in pairs:
    r_before = vectorized_pearson(masked_all[s1],  masked_all[s2])
    r_after  = vectorized_pearson(aligned_all[s1], aligned_all[s2])
    isc_before_pairs.append(r_before)
    isc_after_pairs.append(r_after)
    pair_labels.append((s1, s2))
    print(f"  {s1} x {s2}: mean r before={r_before.mean():.3f}  after={r_after.mean():.3f}")

isc_before_mean = np.mean(isc_before_pairs, axis=0)
isc_after_mean  = np.mean(isc_after_pairs,  axis=0)

print(f"\nOverall mean ISC before: {isc_before_mean.mean():.4f}")
print(f"Overall mean ISC after:  {isc_after_mean.mean():.4f}")

np.save(SAVE_DIR / "isc_before_mean.npy",  isc_before_mean)
np.save(SAVE_DIR / "isc_after_mean.npy",   isc_after_mean)
np.save(SAVE_DIR / "isc_before_pairs.npy", np.array(isc_before_pairs))
np.save(SAVE_DIR / "isc_after_pairs.npy",  np.array(isc_after_pairs))
with open(SAVE_DIR / "pair_labels.pkl", "wb") as f:
    pickle.dump(pair_labels, f)

masker.inverse_transform(isc_before_mean).to_filename(SAVE_DIR / "isc_before_mean.nii.gz")
masker.inverse_transform(isc_after_mean).to_filename(SAVE_DIR / "isc_after_mean.nii.gz")
masker.inverse_transform(isc_after_mean - isc_before_mean).to_filename(SAVE_DIR / "isc_diff.nii.gz")
print(f"Saved ISC arrays and NIfTIs to {SAVE_DIR}")

print("\nDone.")
