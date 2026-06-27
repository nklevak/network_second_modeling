# 23_hyperalignment_surface_run.py
# Template-based hyperalignment on fsaverage6 surface data (.func.gii).
# Analogous to prelim_discovery_analysis/23_hyperalignment_smor_run.py but
# uses surface GIFTI files instead of volumetric NIfTIs.
#
# Key differences from volumetric version:
#   - Loads hemi-L and hemi-R .func.gii files; concatenates into (n_contrasts, 81924) arrays
#   - Uses Schaefer 400 parcels projected onto fsaverage6 for alignment chunks
#   - No NiftiMasker; works directly with 2D numpy arrays
#   - Saves scores as both .npy and .func.gii instead of NIfTI

import os
import sys
import pickle
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
import joblib
from fmralign import GroupAlignment, PairwiseAlignment

sys.path.insert(0, str(Path(__file__).parent.parent / "prelim_discovery_analysis"))
from utils import TASKS, CONTRASTS, SUBJECTS, SESSIONS

# ── Paths ─────────────────────────────────────────────────────────────────────
SURFACE_BASE_DIR = Path('/scratch/groups/russpold/logben/discovery_bids/derivatives/lev1_surface')
ANALYSIS_DIR     = Path(__file__).parent
OUTPUT_DIR       = ANALYSIS_DIR / "hyperalignment_fmralign_results" / "discovery_sample" / "schaefer400_surface"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RT_CONTRAST        = "response_time"
N_VERTICES_PER_HEMI = 40962   # fsaverage6


##########################################################################
# path builder
def build_surface_contrast_map_path(base_dir, subject, session, task, contrast, hemi, file_type="default"):
    """Return Path to a .func.gii contrast map for one hemisphere."""
    file_ending = "z_score" if file_type == "z" else "effect-size"
    filename = (
        f'{subject}_{session}_task-{task}_run-1_hemi-{hemi}'
        f'_contrast-{contrast}_rtmodel-RTDur_stat-{file_ending}.func.gii'
    )
    return Path(base_dir) / subject / f'task-{task}' / 'indiv_contrasts' / filename


##########################################################################
# load beta maps
def load_surface_beta_maps() -> dict:
    """
    Load .func.gii beta maps for all subjects/tasks/contrasts/encounters.
    Both hemispheres are loaded and concatenated into a single (81924,) vector.

    Returns
    -------
    maps : {task: {contrast: {subject: {enc_key: np.ndarray(81924,)}}}}
        Only encounters where BOTH hemi-L and hemi-R files exist are included.
    """
    maps = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

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
                        lh_data = nib.load(lh_path).darrays[0].data   # (40962,)
                        rh_data = nib.load(rh_path).darrays[0].data   # (40962,)
                        maps[task][contrast][subject][enc_key] = np.concatenate([lh_data, rh_data])
                        enc_count += 1

    for task in TASKS:
        for contrast in CONTRASTS[task]:
            for subject in SUBJECTS:
                n_enc = len(maps[task][contrast].get(subject, {}))
                if n_enc > 0:
                    print(f"  {subject} | {task} | {contrast}: {n_enc} encounters")
                else:
                    print(f"  MISSING: {subject} | {task} | {contrast}")

    return maps


##########################################################################
# Schaefer 400 surface labels
def build_schaefer400_surface_labels() -> np.ndarray:
    """
    Project the Schaefer 400 volumetric atlas onto the fsaverage6 surface
    (nearest-neighbour interpolation) to obtain per-vertex parcel labels.

    Returns
    -------
    labels : np.ndarray, shape (81924,), dtype int
        Parcel ID per vertex (LH concatenated with RH).
        Medial wall vertices that receive label 0 are assigned to a
        catch-all parcel (401) so every vertex belongs to a chunk.
    """
    from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_schaefer_2018
    from nilearn.surface import vol_to_surf

    print("Projecting Schaefer 400 atlas onto fsaverage6 surface...")
    schaefer = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=1)
    fsavg6   = fetch_surf_fsaverage('fsaverage6')

    lh_labels = vol_to_surf(
        schaefer.maps, fsavg6.pial_left,  interpolation='nearest_most_frequent', radius=0
    ).astype(int)
    rh_labels = vol_to_surf(
        schaefer.maps, fsavg6.pial_right, interpolation='nearest_most_frequent', radius=0
    ).astype(int)

    labels = np.concatenate([lh_labels, rh_labels])

    # Schaefer 400: LH parcels 1-200, RH parcels 201-400 already distinct.
    # Assign any unassigned (medial wall) vertices to a catch-all chunk.
    n_medial = (labels == 0).sum()
    if n_medial > 0:
        labels[labels == 0] = 401
        print(f"  Assigned {n_medial} medial wall vertices to catch-all parcel 401")

    print(f"  {len(np.unique(labels))} unique parcels across {len(labels)} vertices")
    return labels


##########################################################################
# scoring
def score_vertexwise(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Per-vertex Pearson r between two (n_contrasts, n_vertices) arrays.
    Returns shape (n_vertices,).
    """
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    denom = np.sqrt((A ** 2).sum(0) * (B ** 2).sum(0)) + 1e-10
    return (A * B).sum(0) / denom


##########################################################################
# GIFTI saving helper
def save_as_gifti(score: np.ndarray, out_dir: Path, stem: str) -> None:
    """
    Split a (81924,) score array into LH / RH and save as .func.gii files.
    """
    lh = score[:N_VERTICES_PER_HEMI]
    rh = score[N_VERTICES_PER_HEMI:]
    for hemi, data in [('L', lh), ('R', rh)]:
        darray = nib.gifti.GiftiDataArray(data.astype(np.float32))
        img    = nib.gifti.GiftiImage(darrays=[darray])
        nib.save(img, out_dir / f'{stem}_hemi-{hemi}.func.gii')


##########################################################################
if __name__ == "__main__":
    # ── 1. Load beta maps ──────────────────────────────────────────────────────
    beta_maps = load_surface_beta_maps()

    # ── 2. Find shared (task, contrast, encounter) tuples ─────────────────────
    task_order = {t: i for i, t in enumerate(TASKS)}
    available  = {subj: set() for subj in SUBJECTS}
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
    print(f"Shared (task, contrast, encounter) tuples: {len(shared_tce_sorted)}")

    with open(OUTPUT_DIR / "shared_tce_sorted.pkl", "wb") as f:
        pickle.dump(shared_tce_sorted, f)

    # ── 3. Build per-subject 2D arrays: (n_contrasts, n_vertices) ─────────────
    subject_arrays = {}
    for subj in SUBJECTS:
        rows = [beta_maps[task][contrast][subj][enc]
                for task, contrast, enc in shared_tce_sorted]
        subject_arrays[subj] = np.stack(rows, axis=0).astype(np.float32)
        print(f"  {subj}: {subject_arrays[subj].shape}")
    del beta_maps

    # ── 4. Build Schaefer 400 surface parcel labels ────────────────────────────
    labels = build_schaefer400_surface_labels()

    # ── 5. Split subjects ──────────────────────────────────────────────────────
    train_subjects   = SUBJECTS[:-1]
    left_out_subject = SUBJECTS[-1]

    masked_train  = {s: subject_arrays[s] for s in train_subjects}
    left_out_data = subject_arrays[left_out_subject]

    # ── 6. Baseline: average of training subjects ──────────────────────────────
    euclidean_avg = np.mean(list(masked_train.values()), axis=0)

    # ── 7. Fit group template via Procrustes ───────────────────────────────────
    print("Fitting GroupAlignment (Procrustes)...")
    template_estim = GroupAlignment(method="procrustes", labels=labels)
    template_estim.fit(X=masked_train, y="template")
    procrustes_template = template_estim.template

    # ── 8. Align left-out subject to template ─────────────────────────────────
    print("Fitting PairwiseAlignment for left-out subject...")
    pairwise_estim = PairwiseAlignment(method="procrustes", labels=labels).fit(
        left_out_data, procrustes_template
    )
    predictions_from_template = pairwise_estim.transform(left_out_data)

    # ── 9. Score ───────────────────────────────────────────────────────────────
    average_score  = score_vertexwise(left_out_data, euclidean_avg)
    template_score = score_vertexwise(predictions_from_template, procrustes_template)
    print(f"Baseline mean r (left-out vs. group avg):   {average_score.mean():.4f}")
    print(f"Template mean r (aligned vs. template):     {template_score.mean():.4f}")

    # ── 10. Save ───────────────────────────────────────────────────────────────
    joblib.dump(template_estim, OUTPUT_DIR / "group_alignment.pkl")
    joblib.dump(pairwise_estim, OUTPUT_DIR / "pairwise_alignment.pkl")
    np.save(OUTPUT_DIR / "procrustes_template.npy",         procrustes_template)
    np.save(OUTPUT_DIR / "schaefer400_surface_labels.npy",  labels)
    np.save(OUTPUT_DIR / "average_score.npy",               average_score)
    np.save(OUTPUT_DIR / "template_score.npy",              template_score)
    print(f"Saved alignment estimators and score arrays to {OUTPUT_DIR}")

    save_as_gifti(average_score,  OUTPUT_DIR, "average_score")
    save_as_gifti(template_score, OUTPUT_DIR, "template_score")
    print(f"Saved score GIFTIs to {OUTPUT_DIR}")
