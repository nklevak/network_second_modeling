# using fmralign package to do template-based (not pairwise) hyperalignment
import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from collections import defaultdict
from itertools import combinations
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp, linregress
from statsmodels.stats.multitest import multipletests
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import LabelEncoder
from nilearn.maskers import NiftiMasker
from nilearn.image import concat_imgs, math_img
from nilearn import plotting
from templateflow import api
import joblib
from fmralign import GroupAlignment, PairwiseAlignment
from fmralign.embeddings.parcellation import get_labels
from fmralign.metrics import score_voxelwise
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    TASKS, CONTRASTS, SUBJECTS, SESSIONS, ENCOUNTERS,
    build_first_level_contrast_map_path, is_valid_contrast_map, clean_z_map_data,
    convert_to_regular_dict, create_smor_atlas,load_smor_atlas, load_schaefer_atlas, cleanup_memory
)
from config import OUTPUT_DIRS, BASE_DIR

ANALYSIS_DIR   = Path(__file__).parent
DATA_DIR       = ANALYSIS_DIR / OUTPUT_DIRS["smor"]
OUTPUT_DIR     = ANALYSIS_DIR / "hyperalignment_fmralign_results" / "discovery_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_PARCELS        = 429
ENCOUNTER_CENTER = 3.0   # encounters 1-5 → centered: -2,-1,0,1,2
RT_CONTRAST = "response_time" # going to exclude this one for now

#########################################################################################
# using the betas not the z-scores
def load_parcel_dict(output_ending = "_betas", date_updated = "0111") -> dict:
    """
    Merge all 3 parcellated beta pkl files into one dict.
    Structure: parcel_dict[subject][task][contrast][encounter]
               → DataFrame with columns ['region', 'activation']
    """
    # other output_ending option is: "_z_scored"

    parcel_dict = {}
    mean_filename = f"{main_files['mean']}_{date_updated}" # fname = DATA_DIR / f"discovery_parcel_indiv_mean_updated_{date_updated}_{idx}_betas.pkl"

    for idx in [1, 2, 3]:
        fname = DATA_DIR / f"{mean_filename}_{idx}_betas.pkl"
        if not fname.exists():
            warnings.warn(f"Missing pkl file: {fname}")
            continue
        with open(fname, "rb") as f:
            chunk = pickle.load(f)
        parcel_dict.update(chunk)
        print(f"  Loaded {fname.name} → {len(chunk)} subjects")
    print(f"Total subjects loaded: {list(parcel_dict.keys())}")
    return parcel_dict

def get_activation_vector(parcel_dict, subject, task, contrast, encounter):
    """
    Extract the 429-dim activation vector for one subject/task/contrast/encounter.
    Returns None if not present or wrong length.
    """
    try:
        df = parcel_dict[subject][task][contrast][encounter]
        if df is None:
            return None
        v = df.set_index("region")["activation"].values.astype(float)
        return v if len(v) == N_PARCELS else None
    except (KeyError, AttributeError):
        return None

def build_contrast_matrix(parcel_dict: dict) -> tuple[dict, list]:
    """
    For all subjects build a ((N_sessions x N_shared_contrasts) × 429) matrix.

    Only contrasts/sessions present for ALL subjects are included (shared rows guarantee
    that Procrustes operates on semantically identical rows across subjects).
    response_time contrasts are excluded from the alignment signal.

    Remove any sessions that are not done by all subjects.

    Returns:
        matrices : {subject -> np.ndarray (N_shared × 429)}
        row_labels: list of "task__contrast" strings (same order for all subjects)
    """

    print("\n=== STEP 1: Building contrast matrices ===")

    # Collect all available (task, contrast, enc) tuples per subject
    available: dict[str, set] = {s: set() for s in SUBJECTS}
    for subj in SUBJECTS:
        for task in TASKS:
            for contrast in CONTRASTS[task]:
                if contrast == RT_CONTRAST:
                    continue
                for enc in ENCOUNTERS:
                    v = get_activation_vector(parcel_dict, subj, task, contrast, enc)
                    if v is not None:
                        available[subj].add((task, contrast, enc))
                        # no break — collect ALL valid encounters

    # Shared rows = intersection across all subjects
    shared = set.intersection(*available.values())
    # Sort deterministically: by task order, then contrast order, then encounter
    task_order = {t: i for i, t in enumerate(TASKS)}
    row_labels_sorted = sorted(
        shared,
        key=lambda tce: (task_order[tce[0]], CONTRASTS[tce[0]].index(tce[1]), tce[2])
    )
    row_labels = [f"{t}__{c}__{e}" for t, c, e in row_labels_sorted]

    print(f"Shared (task, contrast, encounter) tuples across all subjects: {len(row_labels_sorted)}")
    for rc in row_labels_sorted:
        print(f"  {rc[0]}/{rc[1]}/enc{rc[2]}")

    # Build per-subject matrices
    matrices = {}
    for subj in SUBJECTS:
        rows = []
        for task, contrast, enc in row_labels_sorted:
            v = get_activation_vector(parcel_dict, subj, task, contrast, enc)
            rows.append(v)
        mat = np.array(rows)                           # (N_shared × 429)
        matrices[subj] = mat
        print(f"  {subj}: contrast matrix shape {mat.shape}")

    return matrices, row_labels

##########################################################################
# load raw beta NIfTI maps (not parcellated)
def load_beta_maps() -> dict:
    """
    Load raw beta NIfTI maps for all subjects/tasks/contrasts/encounters.

    Mirrors the notebook approach in 15_cleaned_parcellate_individuals.ipynb:
    iterates over SESSIONS per subject/task/contrast and counts encounters
    independently for each combination.

    Returns:
        maps : {task: {contrast: {subject: {encounter_str: NIfTI image}}}}
                encounter_str is zero-padded, e.g. '01', '02', ..., '05'
    """
    maps = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for task in TASKS:
        for contrast in CONTRASTS[task]:
            for subject in SUBJECTS:
                overall_encounter_count = 0
                for session in SESSIONS:
                    contrast_map_path = build_first_level_contrast_map_path(
                        BASE_DIR, subject, session, task, contrast
                    )
                    if os.path.exists(contrast_map_path):
                        enc_key = f"{overall_encounter_count + 1:02d}"
                        maps[task][contrast][subject][enc_key] = nib.load(contrast_map_path)
                        overall_encounter_count += 1

    # Report what was loaded
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
# use fmr align to make a group template and then do procrustes on it to aign all the subjects:
# from this tutorial: https://fmralign.github.io/fmralign/auto_examples/plot_group_alignment.html

if __name__ == "__main__":
    # Load raw beta NIfTI maps
    beta_maps = load_beta_maps()

    # Find shared (task, contrast, encounter) tuples present for ALL subjects in beta_maps
    # (required so every subject's 4D image has identical volume ordering for Procrustes alignment)
    task_order = {t: i for i, t in enumerate(TASKS)}
    available = {subj: set() for subj in SUBJECTS}
    for subj in SUBJECTS:
        for task in TASKS:
            for contrast in CONTRASTS[task]:
                if contrast == RT_CONTRAST:
                    continue
                for enc_key, img in beta_maps[task][contrast].get(subj, {}).items():
                    available[subj].add((task, contrast, enc_key))
    shared_tce = set.intersection(*available.values())
    shared_tce_sorted = sorted(
        shared_tce,
        key=lambda tce: (task_order[tce[0]], CONTRASTS[tce[0]].index(tce[1]), tce[2])
    )
    print(f"Shared (task, contrast, encounter) tuples across all subjects: {len(shared_tce_sorted)}")

    # Build per-subject 4D image using only shared tuples (excluding RT)
    subject_imgs = {}
    for subj in SUBJECTS:
        imgs = [beta_maps[task][contrast][subj][enc] for task, contrast, enc in shared_tce_sorted]
        subject_imgs[subj] = concat_imgs(imgs)
    del beta_maps

    # Set up gray matter mask and masker (GM prob > 0.2 implicitly excludes non-brain voxels)
    gm_prob = api.get('MNI152NLin2009cAsym', label='GM', suffix='probseg', resolution=2)
    masker = NiftiMasker(mask_img=math_img("gm > 0.2", gm=gm_prob)).fit()

    # Split into training subjects and left-out subject (SUBJECTS[-1])
    train_subjects = SUBJECTS[:-1]
    left_out_subject = SUBJECTS[-1]
    template_train = [subject_imgs[s] for s in train_subjects]  # NIfTIs (for get_labels)

    # Mask all subjects to 2D arrays (required by GroupAlignment and for scoring)
    masked_train = {s: masker.transform(subject_imgs[s]) for s in train_subjects}
    left_out_data = masker.transform(subject_imgs[left_out_subject])

    # Compute baseline (voxelwise average of training subjects in masked space)
    euclidean_avg = np.mean(list(masked_train.values()), axis=0)  # (n_vols, n_voxels)

    # Create parcel labels from the first training image (get_labels expects NIfTI)
    labels = get_labels(template_train[0], n_pieces=500, masker=masker)

    # Fit group template using Procrustes alignment (GroupAlignment expects 2D arrays)
    template_estim = GroupAlignment(method="procrustes", labels=labels)
    template_estim.fit(X=masked_train, y="template")
    procrustes_template = template_estim.template  # 2D array (n_vols, n_voxels)

    # Align left-out subject to the template (both 2D arrays)
    pairwise_estim = PairwiseAlignment(method="procrustes", labels=labels).fit(
        left_out_data, procrustes_template
    )
    predictions_from_template = pairwise_estim.transform(left_out_data)  # 2D array

    # Score: score_voxelwise expects NIfTIs + masker — inverse_transform 2D arrays first
    # Baseline: how well does the unaligned group average match the left-out subject?
    # Template: how well does the aligned left-out subject match the group template?
    euclidean_avg_img = masker.inverse_transform(euclidean_avg)
    predictions_img = masker.inverse_transform(predictions_from_template)
    procrustes_template_img = masker.inverse_transform(procrustes_template)
    left_out_img = subject_imgs[left_out_subject]

    average_score = score_voxelwise(left_out_img, euclidean_avg_img, masker, loss="corr")
    template_score = score_voxelwise(predictions_img, procrustes_template_img, masker, loss="corr")

    # Save results
    SAVE_DIR = OUTPUT_DIR / "ward_500pieces"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Save score NIfTIs
    average_score_img = masker.inverse_transform(average_score)
    template_score_img = masker.inverse_transform(template_score)
    average_score_img.to_filename(SAVE_DIR / "average_score.nii.gz")
    template_score_img.to_filename(SAVE_DIR / "template_score.nii.gz")
    print(f"Saved score NIfTIs to {SAVE_DIR}")

    # Save plots
    baseline_display = plotting.plot_stat_map(
        average_score_img, display_mode="z", vmax=1, cut_coords=[-15, -5]
    )
    baseline_display.title("Left-out subject correlation with group average")
    baseline_display.savefig(SAVE_DIR / "average_score.png")

    display = plotting.plot_stat_map(
        template_score_img, display_mode="z", cut_coords=[-15, -5], vmax=1
    )
    display.title("Aligned subject correlation with Procrustes template")
    display.savefig(SAVE_DIR / "template_score.png")
    print(f"Saved plots to {SAVE_DIR}")

    # Save alignment estimators and template
    joblib.dump(template_estim, SAVE_DIR / "group_alignment.pkl")
    joblib.dump(pairwise_estim, SAVE_DIR / "pairwise_alignment.pkl")
    np.save(SAVE_DIR / "procrustes_template.npy", procrustes_template)
    print(f"Saved alignment estimators to {SAVE_DIR}")

