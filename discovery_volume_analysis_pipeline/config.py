"""
config.py
=========
Central configuration for the discovery volume hyperalignment pipeline.
All run/analysis scripts import from here so paths and constants stay in one place.
"""

import sys
import numpy as np
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR  = ANALYSIS_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# First-level volumetric contrast maps
BASE_DIR = Path(
    '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_20250402/'
    'derivatives/archive/dataset-networkDiscovery_model-lev1_space-MNI_'
    'withinMaskThreshold-1.0_rtmodel-RTDur/'
)

# Import shared study constants from the existing prelim pipeline
sys.path.insert(0, str(ANALYSIS_DIR.parent / "prelim_discovery_analysis"))
from utils import (
    TASKS, CONTRASTS, SUBJECTS, SESSIONS, ENCOUNTERS,
    build_first_level_contrast_map_path,
)

# ── Analysis constants ─────────────────────────────────────────────────────────
RT_CONTRAST      = "response_time"
ENCOUNTER_CENTER = 3.0   # center of [1..5]; used for practice-slope regression

# One representative contrast per task — used for RSA RDMs
RSA_CONTRASTS = {
    "stopSignal":         "stop_failure-go",
    "goNogo":             "nogo_success-go",
    "nBack":              "twoBack-oneBack",
    "directedForgetting": "neg-con",
    "cuedTS":             "task_switch_cue_switch-task_stay_cue_stay",
    "spatialTS":          "task_switch_cue_switch-task_stay_cue_stay",
    "shapeMatching":      "main_vars",
    "flanker":            "incongruent-congruent",
}

# ── Atlas / masker setup ───────────────────────────────────────────────────────
def setup_masker_and_labels(reference_img):
    """
    Fetch Schaefer 400 atlas (2 mm MNI), resample to reference_img's voxel grid,
    build a NiftiMasker over the atlas footprint, and extract a 1-D label array
    (one integer parcel ID per masked voxel).

    Parameters
    ----------
    reference_img : Nifti1Image
        Any single contrast map from the dataset.  Its affine and shape define
        the target voxel grid.

    Returns
    -------
    masker : NiftiMasker
        Fitted masker; use masker.transform() / masker.inverse_transform().
    labels : np.ndarray, shape (n_voxels,)
        Integer parcel IDs 1–400, one per masked voxel.
    """
    from nilearn.datasets import fetch_atlas_schaefer_2018
    from nilearn.image import resample_to_img, math_img
    from nilearn.maskers import NiftiMasker

    schaefer = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2)
    atlas_resampled = resample_to_img(
        schaefer.maps, reference_img,
        interpolation="nearest", force_resample=True, copy_header=True,
    )
    mask_img = math_img("atlas > 0", atlas=atlas_resampled)
    masker   = NiftiMasker(mask_img=mask_img).fit()
    labels   = masker.transform(atlas_resampled).astype(int).ravel()

    n_voxels  = labels.shape[0]
    n_parcels = len(np.unique(labels))
    print(f"Masker: {n_voxels} voxels | {n_parcels} Schaefer-400 parcels "
          f"(avg {n_voxels / n_parcels:.0f} voxels/parcel)")
    return masker, labels


# ── Data loading ───────────────────────────────────────────────────────────────
def load_all_masked_data(masker, verbose=True):
    """
    Load and mask every available non-RT contrast map for every subject,
    task, contrast, and encounter.

    Returns
    -------
    all_masked : dict
        Keys are (subject, task, contrast, encounter_str) 4-tuples.
        Values are np.ndarray of shape (n_voxels,) in float32.
    """
    import nibabel as nib

    all_masked = {}
    for task in TASKS:
        for contrast in CONTRASTS[task]:
            if contrast == RT_CONTRAST:
                continue
            for subject in SUBJECTS:
                enc_count = 0
                for session in SESSIONS:
                    path = build_first_level_contrast_map_path(
                        BASE_DIR, subject, session, task, contrast
                    )
                    if Path(path).exists():
                        enc_key = f"{enc_count + 1:02d}"
                        img = nib.load(path)
                        all_masked[(subject, task, contrast, enc_key)] = (
                            masker.transform(img).ravel().astype(np.float32)
                        )
                        enc_count += 1
        if verbose:
            print(f"  Loaded: {task}")
    return all_masked


def find_shared_tce(all_masked, subjects, tasks_subset=None, encounters_subset=None):
    """
    Find (task, contrast, encounter) tuples present for ALL subjects,
    optionally restricted to a subset of tasks or encounters.

    Returns a sorted list of tuples in canonical order
    (by task index, contrast index within task, encounter string).
    """
    task_order     = {t: i for i, t in enumerate(TASKS)}
    tasks_to_check = tasks_subset if tasks_subset is not None else TASKS
    encs_to_check  = set(encounters_subset) if encounters_subset is not None else set(ENCOUNTERS)

    available = {s: set() for s in subjects}
    for (subj, task, contrast, enc), _ in all_masked.items():
        if subj not in subjects:
            continue
        if task not in tasks_to_check:
            continue
        if enc not in encs_to_check:
            continue
        available[subj].add((task, contrast, enc))

    shared = set.intersection(*available.values())
    return sorted(
        shared,
        key=lambda tce: (task_order[tce[0]], CONTRASTS[tce[0]].index(tce[1]), tce[2]),
    )


def build_subject_arrays(all_masked, subjects, tce_list):
    """
    Stack masked voxel arrays into (n_contrasts, n_voxels) matrices.

    Parameters
    ----------
    all_masked : dict  — from load_all_masked_data()
    subjects   : list of subject IDs
    tce_list   : list of (task, contrast, enc) tuples defining row order

    Returns
    -------
    dict  {subject_id: np.ndarray (n_contrasts, n_voxels)}
    """
    return {
        subj: np.stack([all_masked[(subj, t, c, e)] for t, c, e in tce_list])
        for subj in subjects
    }
