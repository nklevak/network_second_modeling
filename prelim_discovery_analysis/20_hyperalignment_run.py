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
from nilearn.image import concat_imgs
from fmralign import GroupAlignment
from fmralign.embeddings.parcellation import get_labels
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
OUTPUT_DIR.mkdir(exist_ok=True)

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
    # get the subject data:
    beta_contrast_dict = load_beta_maps()
    contrast_matrices = build_contrast_matrix(beta_contrast_dict)

    # set up a masker
    masker = NiftiMasker(mask_img=mask_img).fit()

    # get all the training samples (here let SUBJECTS[-1] be the held out subject)
    template_train = []
    for i in range(len(SUBJECTS) - 1):
        template_train.append(concat_imgs(contrast_matrices[i]))
    left_out_subject = concat_imgs(contrast_matrices[SUBJECTS[-1]])

    # compute baseline (average of subjects):
    # masked_imgs = [masker.transform(img) for img in template_train] # not masking for now
    # euclidean_avg = np.mean(masked_imgs, axis=0)
    euclidean_avg = np.mean(template_train, axis=0)



