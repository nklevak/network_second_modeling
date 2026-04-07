"""
Hyperalignment pipeline for neural practice effects.
Methods: Condition/Contrast-Level Procrustes (Method A from 20_hyperalignment_plan.py)

Steps:
    1. Build session-averaged contrast matrices per subject (N_contrasts × 429)
    2. Iterative Procrustes hyperalignment → rotation R_i per subject
    3. Apply R_i to every session-level beta map
    4. LME per parcel in common space: activation ~ encounter_c + (1+encounter_c|subject)
    5. Validate alignment via ISC and cross-subject decoding

References:
    Haxby et al. 2011 Neuron; Bazeille et al. 2021 NeuroImage;
    Zhang et al. 2020 IEEE TCDS; Chen et al. 2013 NeuroImage
"""

import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
from scipy.stats import ttest_1samp, linregress
from statsmodels.stats.multitest import multipletests
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent))
from utils import TASKS, CONTRASTS, SUBJECTS, ENCOUNTERS
from config import OUTPUT_DIRS

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

ANALYSIS_DIR   = Path(__file__).parent
DATA_DIR       = ANALYSIS_DIR / OUTPUT_DIRS["smor"]
OUTPUT_DIR     = ANALYSIS_DIR / "hyperalignment_results"
OUTPUT_DIR.mkdir(exist_ok=True)

N_PARCELS       = 429
ENCOUNTER_CENTER = 3.0   # encounters 1-5 → centered: -2,-1,0,1,2
N_PROCRUSTES_ITER = 30
PROCRUSTES_TOL    = 1e-7

# Contrasts to use for alignment (exclude response_time — less interpretable
# as a cognitive condition label; keep task-meaningful contrasts only).
# Subjects must have data for a row for it to be included in alignment.
RT_CONTRAST = "response_time"

# ---------------------------------------------------------------------------
# STEP 0: Data loading
# ---------------------------------------------------------------------------

def load_parcel_dict() -> dict:
    """
    Merge all 3 parcellated beta pkl files into one dict.
    Structure: parcel_dict[subject][task][contrast][encounter]
               → DataFrame with columns ['region', 'activation']
    """
    parcel_dict = {}
    for idx in [1, 2, 3]:
        fname = DATA_DIR / f"discovery_parcel_indiv_mean_updated_0111_{idx}_betas.pkl"
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


# ---------------------------------------------------------------------------
# STEP 1: Build session-averaged contrast matrix per subject
# ---------------------------------------------------------------------------

def build_contrast_matrices(parcel_dict: dict) -> tuple[dict, list]:
    """
    For each subject build a (N_shared_contrasts × 429) matrix by averaging
    beta maps across available encounters per contrast.

    Only contrasts present for ALL subjects are included (shared rows guarantee
    that Procrustes operates on semantically identical rows across subjects).
    response_time contrasts are excluded from the alignment signal.

    Returns:
        matrices : {subject -> np.ndarray (N_shared × 429)}
        row_labels: list of "task__contrast" strings (same order for all subjects)
    """
    print("\n=== STEP 1: Building contrast matrices ===")

    # Collect all available (task, contrast) pairs per subject
    available: dict[str, set] = {s: set() for s in SUBJECTS}
    for subj in SUBJECTS:
        for task in TASKS:
            for contrast in CONTRASTS[task]:
                if contrast == RT_CONTRAST:
                    continue
                # Check at least one encounter exists
                for enc in ENCOUNTERS:
                    v = get_activation_vector(parcel_dict, subj, task, contrast, enc)
                    if v is not None:
                        available[subj].add((task, contrast))
                        break

    # Shared rows = intersection across all subjects
    shared = set.intersection(*available.values())
    # Sort deterministically: by task order then contrast order
    task_order = {t: i for i, t in enumerate(TASKS)}
    row_labels_sorted = sorted(
        shared,
        key=lambda tc: (task_order[tc[0]], CONTRASTS[tc[0]].index(tc[1]))
    )
    row_labels = [f"{t}__{c}" for t, c in row_labels_sorted]

    print(f"Shared (task, contrast) pairs across all subjects: {len(row_labels_sorted)}")
    for rc in row_labels_sorted:
        print(f"  {rc[0]}/{rc[1]}")

    # Build per-subject matrices
    matrices = {}
    for subj in SUBJECTS:
        rows = []
        for task, contrast in row_labels_sorted:
            enc_vecs = [
                get_activation_vector(parcel_dict, subj, task, contrast, enc)
                for enc in ENCOUNTERS
                if get_activation_vector(parcel_dict, subj, task, contrast, enc) is not None
            ]
            rows.append(np.mean(enc_vecs, axis=0))   # average across encounters
        mat = np.array(rows)                           # (N_shared × 429)
        matrices[subj] = mat
        print(f"  {subj}: contrast matrix shape {mat.shape}")

    return matrices, row_labels


# ---------------------------------------------------------------------------
# STEP 2: Iterative Procrustes hyperalignment
# ---------------------------------------------------------------------------

def iterative_procrustes(matrices: dict) -> tuple[dict, dict, np.ndarray]:
    """
    Iterative group Procrustes hyperalignment.

    For each iteration:
        1. Find orthogonal rotation R_i = argmin ||X_i @ R_i - T||_F  (scipy)
        2. Update template T = mean(X_i @ R_i)

    All subjects must have matrices with the same number of rows (guaranteed by
    build_contrast_matrices).

    Returns:
        rotations : {subject -> R (429 × 429)}
        aligned   : {subject -> aligned matrix (N_shared × 429)}
        template  : final group template (N_shared × 429)
    """
    print("\n=== STEP 2: Iterative Procrustes hyperalignment ===")

    subjects = list(matrices.keys())
    template = np.mean(np.stack(list(matrices.values())), axis=0)

    rotations, aligned = {}, {}
    for it in range(N_PROCRUSTES_ITER):
        rotations, aligned = {}, {}
        for subj in subjects:
            R, _ = orthogonal_procrustes(matrices[subj], template)
            rotations[subj] = R
            aligned[subj]   = matrices[subj] @ R
        new_template = np.mean(np.stack(list(aligned.values())), axis=0)
        change = np.linalg.norm(new_template - template)
        template = new_template
        print(f"  Iter {it+1:02d}: template change = {change:.2e}")
        if change < PROCRUSTES_TOL:
            print(f"  Converged at iteration {it+1}.")
            break
    else:
        print(f"  Reached max iterations (final change={change:.2e}).")

    # Report alignment quality: mean ISC of aligned vs unaligned contrast matrices
    def mean_pairwise_isc(mats):
        vecs = [mats[s].flatten() for s in subjects]
        rs = [pearsonr(vecs[i], vecs[j])[0] for i, j in combinations(range(len(vecs)), 2)]
        return np.mean(rs)

    isc_before = mean_pairwise_isc(matrices)
    isc_after  = mean_pairwise_isc(aligned)
    print(f"\n  Mean pairwise ISC on contrast matrices:")
    print(f"    Before alignment: {isc_before:.4f}")
    print(f"    After  alignment: {isc_after:.4f}  (Δ = {isc_after - isc_before:+.4f})")

    return rotations, aligned, template


# ---------------------------------------------------------------------------
# STEP 3: Apply rotation to all session-level beta maps
# ---------------------------------------------------------------------------

def apply_rotations(parcel_dict: dict, rotations: dict) -> dict:
    """
    Apply each subject's rotation matrix to every session-level beta vector.

    Returns:
        aligned_data : {subject -> {task -> {contrast -> {encounter -> np.ndarray(429,)}}}}
    """
    print("\n=== STEP 3: Applying rotations to session-level data ===")

    aligned_data = {}
    for subj in SUBJECTS:
        R = rotations[subj]
        aligned_data[subj] = {}
        n_aligned = 0
        for task in TASKS:
            aligned_data[subj][task] = {}
            for contrast in CONTRASTS[task]:
                aligned_data[subj][task][contrast] = {}
                for enc in ENCOUNTERS:
                    v = get_activation_vector(parcel_dict, subj, task, contrast, enc)
                    if v is not None:
                        aligned_data[subj][task][contrast][enc] = v @ R
                        n_aligned += 1
        print(f"  {subj}: aligned {n_aligned} session beta vectors")

    return aligned_data


# ---------------------------------------------------------------------------
# STEP 4: Two-stage practice effects per parcel
#   Stage 1: per-subject OLS  activation ~ encounter_c  → slope per subject
#   Stage 2: one-sample t-test across subjects (H0: mean slope = 0)
#   FDR correction across 429 parcels per task/contrast
#
# This replaces LME. With N=5 subjects, LME variance-component estimation
# is unreliable (boundary solutions, failed convergence). The two-stage
# summary-statistics approach is the standard neuroimaging second-level
# method (cf. SPM/FSL) and is equivalent to LME under balanced designs.
# ---------------------------------------------------------------------------

def run_practice_effects(
    aligned_data: dict,
    tasks_to_run=None,
) -> dict:
    """
    For each task/contrast/parcel:
        Stage 1 — fit OLS per subject: activation ~ encounter_c
        Stage 2 — one-sample t-test on the 5 per-subject slopes vs 0
        FDR-BH correction across 429 parcels.

    Returns:
        results[task][contrast] = DataFrame with columns:
            parcel_idx, fixed_slope (mean slope), fixed_slope_p (t-test p),
            t_stat, fdr_significant, n_subjects
    """
    print("\n=== STEP 4: Two-stage practice effects in aligned space ===")

    if tasks_to_run is None:
        tasks_to_run = TASKS

    enc_centered = {enc: (i + 1) - ENCOUNTER_CENTER
                    for i, enc in enumerate(ENCOUNTERS)}

    all_results = {}
    for task in tasks_to_run:
        all_results[task] = {}
        for contrast in CONTRASTS[task]:

            # Stage 1: per-subject OLS slope for each parcel
            # subject_slopes[subj] = (N_PARCELS,) array of slopes
            subject_slopes = {}
            for subj in SUBJECTS:
                enc_vecs = []
                enc_vals = []
                for enc in ENCOUNTERS:
                    vec = aligned_data[subj][task][contrast].get(enc)
                    if vec is not None:
                        enc_vecs.append(vec)           # (N_PARCELS,)
                        enc_vals.append(enc_centered[enc])

                if len(enc_vecs) < 3:                  # need ≥3 points for OLS
                    continue

                X = np.array(enc_vals)                 # (n_enc,)
                Y = np.array(enc_vecs)                 # (n_enc, N_PARCELS)
                # Vectorised OLS: slope = cov(X,Y) / var(X)
                X_c = X - X.mean()
                slopes = (X_c @ Y) / (X_c @ X_c)      # (N_PARCELS,)
                subject_slopes[subj] = slopes

            n_subj = len(subject_slopes)
            if n_subj < 2:
                print(f"  {task}/{contrast}: insufficient subjects ({n_subj}), skipping")
                continue

            # Stage 2: one-sample t-test across subjects per parcel
            slopes_matrix = np.stack(list(subject_slopes.values()))  # (n_subj, N_PARCELS)
            t_stats, p_vals = ttest_1samp(slopes_matrix, popmean=0, axis=0)
            mean_slopes = slopes_matrix.mean(axis=0)

            # FDR correction — first return value of multipletests is the reject array
            valid = ~np.isnan(p_vals)
            fdr_sig = np.zeros(N_PARCELS, dtype=bool)
            if valid.sum() > 0:
                reject, _, _, _ = multipletests(
                    p_vals[valid], alpha=0.05, method="fdr_bh"
                )
                fdr_sig[valid] = reject

            res_df = pd.DataFrame({
                "parcel_idx":      np.arange(N_PARCELS),
                "fixed_slope":     mean_slopes,        # same name → plots unchanged
                "fixed_slope_p":   p_vals,
                "t_stat":          t_stats,
                "fdr_significant": fdr_sig,
                "n_subjects":      n_subj,
            })

            n_sig = fdr_sig.sum()
            print(f"  {task}/{contrast}: {n_subj} subjects, "
                  f"{n_sig}/{N_PARCELS} FDR-sig parcels ({100*n_sig/N_PARCELS:.1f}%)")

            all_results[task][contrast] = res_df

    return all_results


# ---------------------------------------------------------------------------
# STEP 5a: ISC before vs after alignment
# ---------------------------------------------------------------------------

def compute_isc_by_encounter(data: dict, label: str = "") -> pd.DataFrame:
    """
    For each task/contrast/encounter, compute mean pairwise ISC
    across the 429-parcel beta vectors.

    Returns DataFrame with columns: task, contrast, encounter, mean_isc, n_pairs
    """
    print(f"\n=== STEP 5a: ISC by encounter ({label}) ===")

    rows = []
    for task in TASKS:
        for contrast in CONTRASTS[task]:
            for enc_idx, enc in enumerate(ENCOUNTERS):
                vecs = []
                for subj in SUBJECTS:
                    v = data[subj][task][contrast].get(enc)
                    if v is not None:
                        vecs.append(v)
                if len(vecs) < 2:
                    continue
                pairwise_r = [
                    pearsonr(vecs[i], vecs[j])[0]
                    for i, j in combinations(range(len(vecs)), 2)
                ]
                rows.append({
                    "task":       task,
                    "contrast":   contrast,
                    "encounter":  enc_idx + 1,
                    "mean_isc":   np.mean(pairwise_r),
                    "n_pairs":    len(pairwise_r),
                })

    df = pd.DataFrame(rows)
    # Summarise: average ISC per encounter across all contrasts
    summary = df.groupby("encounter")["mean_isc"].mean()
    for enc, isc in summary.items():
        print(f"  Encounter {enc}: mean ISC = {isc:.4f}")

    return df


def build_raw_data_dict(parcel_dict: dict) -> dict:
    """Convert raw parcel_dict into the same {subj/task/contrast/enc -> array} format."""
    raw = {}
    for subj in SUBJECTS:
        raw[subj] = {}
        for task in TASKS:
            raw[subj][task] = {}
            for contrast in CONTRASTS[task]:
                raw[subj][task][contrast] = {}
                for enc in ENCOUNTERS:
                    v = get_activation_vector(parcel_dict, subj, task, contrast, enc)
                    if v is not None:
                        raw[subj][task][contrast][enc] = v
    return raw


# ---------------------------------------------------------------------------
# STEP 5b: Cross-subject decoding before vs after alignment
# ---------------------------------------------------------------------------

def cross_subject_decoding(data: dict, label: str = "") -> pd.DataFrame:
    """
    Leave-one-subject-out cross-subject decoding per encounter.

    Classifier: linear SVM.
    Features:   429-parcel activation vector.
    Labels:     contrast name (predicts which cognitive condition this is).
    Groups:     subject (left out one at a time).

    A higher accuracy after alignment means the aligned space better
    separates cognitive conditions consistently across subjects.

    Returns DataFrame with columns: encounter, mean_accuracy, std_accuracy
    """
    print(f"\n=== STEP 5b: Cross-subject decoding ({label}) ===")

    logo = LeaveOneGroupOut()
    results = []

    for enc_idx, enc in enumerate(ENCOUNTERS):
        X_all, y_all, groups = [], [], []

        for subj in SUBJECTS:
            for task in TASKS:
                for contrast in CONTRASTS[task]:
                    v = data[subj][task][contrast].get(enc)
                    if v is not None:
                        X_all.append(v)
                        y_all.append(f"{task}__{contrast}")
                        groups.append(subj)

        if len(set(groups)) < 2:
            print(f"  Encounter {enc_idx+1}: insufficient subjects, skipping")
            continue

        X   = np.array(X_all)
        y   = np.array(y_all)
        grp = np.array(groups)

        # Encode labels
        le  = LabelEncoder()
        y_enc = le.fit_transform(y)

        clf    = SVC(kernel="linear", C=1.0)
        scores = cross_val_score(clf, X, y_enc, groups=grp, cv=logo, scoring="accuracy")

        mean_acc = scores.mean()
        std_acc  = scores.std()
        print(f"  Encounter {enc_idx+1}: accuracy = {mean_acc:.3f} ± {std_acc:.3f}  (n_folds={len(scores)})")

        results.append({
            "encounter":       enc_idx + 1,
            "mean_accuracy":   mean_acc,
            "std_accuracy":    std_acc,
            "label":           label,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # --- Load data ---
    print("Loading parcel data...")
    parcel_dict = load_parcel_dict()

    # --- Step 1 ---
    contrast_matrices, row_labels = build_contrast_matrices(parcel_dict)

    # --- Step 2 ---
    rotations, aligned_avg_mats, template = iterative_procrustes(contrast_matrices)

    # Save rotations
    rot_path = OUTPUT_DIR / "procrustes_rotations.pkl"
    with open(rot_path, "wb") as f:
        pickle.dump({"rotations": rotations, "row_labels": row_labels, "template": template}, f)
    print(f"\n  Rotations saved → {rot_path}")

    # --- Step 3 ---
    aligned_data = apply_rotations(parcel_dict, rotations)

    aligned_path = OUTPUT_DIR / "aligned_session_data.pkl"
    with open(aligned_path, "wb") as f:
        pickle.dump(aligned_data, f)
    print(f"  Aligned session data saved → {aligned_path}")

    # --- Step 4: Two-stage practice effects ---
    lme_results = run_practice_effects(aligned_data)

    lme_path = OUTPUT_DIR / "lme_results_aligned.pkl"
    with open(lme_path, "wb") as f:
        pickle.dump(lme_results, f)
    print(f"\n  Practice effect results saved → {lme_path}")

    # --- Step 5a: ISC ---
    raw_data    = build_raw_data_dict(parcel_dict)
    isc_before  = compute_isc_by_encounter(raw_data,    label="raw")
    isc_after   = compute_isc_by_encounter(aligned_data, label="aligned")

    isc_combined = pd.concat([isc_before, isc_after], ignore_index=True)
    isc_combined["aligned"] = isc_combined.apply(
        lambda r: "aligned" if r.name >= len(isc_before) else "raw", axis=1
    )
    isc_path = OUTPUT_DIR / "isc_by_encounter.csv"
    isc_combined.to_csv(isc_path, index=False)
    print(f"\n  ISC results saved → {isc_path}")

    # --- Step 5b: Cross-subject decoding ---
    decoding_before = cross_subject_decoding(raw_data,     label="raw")
    decoding_after  = cross_subject_decoding(aligned_data, label="aligned")

    decoding_combined = pd.concat([decoding_before, decoding_after], ignore_index=True)
    dec_path = OUTPUT_DIR / "decoding_by_encounter.csv"
    decoding_combined.to_csv(dec_path, index=False)
    print(f"\n  Decoding results saved → {dec_path}")

    print("\n=== Done ===")
    print(f"All outputs in: {OUTPUT_DIR}")
