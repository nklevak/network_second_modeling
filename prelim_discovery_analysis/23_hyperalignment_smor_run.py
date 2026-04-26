# using fmralign package to do template-based (not pairwise) hyperalignment
# same as 20_hyperalignment_run.py but uses smorgasbord atlas parcels
# instead of Ward parcellation for alignment chunks
import os
import sys
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict
from nilearn.maskers import NiftiMasker
from nilearn.image import concat_imgs, math_img, resample_to_img
from nilearn import plotting
import joblib
from fmralign import GroupAlignment, PairwiseAlignment
from fmralign.metrics import score_voxelwise
sys.path.insert(0, str(Path(__file__).parent))
from utils import (
    TASKS, CONTRASTS, SUBJECTS, SESSIONS,
    build_first_level_contrast_map_path,
)
from config import BASE_DIR

ANALYSIS_DIR = Path(__file__).parent
OUTPUT_DIR   = ANALYSIS_DIR / "hyperalignment_fmralign_results" / "discovery_sample"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMOR_ATLAS_NII = ANALYSIS_DIR / "smorgasbord_atlas_files" / "tpl-MNI152NLin2009cAsym_res-01_atlas-smorgasbord_dseg.nii"

RT_CONTRAST = "response_time"

##########################################################################
# load raw beta NIfTI maps
def load_beta_maps() -> dict:
    """
    Load raw beta NIfTI maps for all subjects/tasks/contrasts/encounters.
    Returns:
        maps : {task: {contrast: {subject: {encounter_str: NIfTI image}}}}
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
# build atlas-based parcel labels aligned to masked voxel space
def build_smor_labels(atlas_resampled, masker):
    """
    Given an already-resampled atlas image and a fitted masker,
    return a 1D integer array with one parcel ID per masked voxel.

    Every voxel included by the masker is guaranteed to have a nonzero
    label because the masker is built from the atlas footprint itself
    (see main block below).
    """
    labels = masker.transform(atlas_resampled).astype(int).ravel()
    print(f"Atlas labels: {len(np.unique(labels))} unique parcels across {len(labels)} voxels")
    return labels


##########################################################################
if __name__ == "__main__":
    # ── 1. Load beta maps ────────────────────────────────────────────────
    beta_maps = load_beta_maps()

    # ── 2. Find shared (task, contrast, encounter) tuples ───────────────
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
    print(f"Shared (task, contrast, encounter) tuples across all subjects: {len(shared_tce_sorted)}")

    # ── 3. Build per-subject 4D images ───────────────────────────────────
    subject_imgs = {}
    for subj in SUBJECTS:
        imgs = [beta_maps[task][contrast][subj][enc] for task, contrast, enc in shared_tce_sorted]
        subject_imgs[subj] = concat_imgs(imgs)
    del beta_maps

    # ── 4. Build masker from smorgasbord atlas footprint ─────────────────
    # Use the atlas itself as the mask so every masked voxel has a label.
    # Use a lightweight 3D reference (first volume) to avoid loading full 4D data.
    from nilearn.image import index_img
    reference_img = index_img(subject_imgs[SUBJECTS[0]], 0)

    atlas_resampled_for_mask = resample_to_img(
        nib.load(SMOR_ATLAS_NII), reference_img,
        interpolation="nearest", force_resample=True, copy_header=True,
    )

    # mask = atlas footprint only; smorgasbord parcels are already all GM
    # (cortical Schaefer + subcortical nuclei + amygdala), no WM included
    mask_img = math_img("atlas > 0", atlas=atlas_resampled_for_mask)
    masker = NiftiMasker(mask_img=mask_img).fit()
    print(f"Mask covers {masker.mask_img_.get_fdata().sum():.0f} voxels")

    # ── 5. Build atlas-derived parcel labels ─────────────────────────────
    labels = build_smor_labels(atlas_resampled_for_mask, masker)

    # ── 6. Split subjects ────────────────────────────────────────────────
    train_subjects   = SUBJECTS[:-1]
    left_out_subject = SUBJECTS[-1]

    # ── 7. Mask all subjects to 2D arrays ────────────────────────────────
    masked_train   = {s: masker.transform(subject_imgs[s]) for s in train_subjects}
    left_out_data  = masker.transform(subject_imgs[left_out_subject])

    # ── 8. Baseline: voxelwise average of training subjects ──────────────
    euclidean_avg = np.mean(list(masked_train.values()), axis=0)  # (n_vols, n_voxels)

    # ── 9. Fit group template via Procrustes ─────────────────────────────
    template_estim = GroupAlignment(method="procrustes", labels=labels)
    template_estim.fit(X=masked_train, y="template")
    procrustes_template = template_estim.template  # (n_vols, n_voxels)

    # ── 10. Align left-out subject to template ───────────────────────────
    pairwise_estim = PairwiseAlignment(method="procrustes", labels=labels).fit(
        left_out_data, procrustes_template
    )
    predictions_from_template = pairwise_estim.transform(left_out_data)

    # ── 11. Score ────────────────────────────────────────────────────────
    euclidean_avg_img     = masker.inverse_transform(euclidean_avg)
    predictions_img       = masker.inverse_transform(predictions_from_template)
    procrustes_template_img = masker.inverse_transform(procrustes_template)
    left_out_img          = subject_imgs[left_out_subject]

    average_score  = score_voxelwise(left_out_img, euclidean_avg_img, masker, loss="corr")
    template_score = score_voxelwise(predictions_img, procrustes_template_img, masker, loss="corr")

    # ── 12. Save ─────────────────────────────────────────────────────────
    SAVE_DIR = OUTPUT_DIR / "smorgasbord_parcels"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    average_score_img  = masker.inverse_transform(average_score)
    template_score_img = masker.inverse_transform(template_score)
    average_score_img.to_filename(SAVE_DIR / "average_score.nii.gz")
    template_score_img.to_filename(SAVE_DIR / "template_score.nii.gz")
    print(f"Saved score NIfTIs to {SAVE_DIR}")

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

    joblib.dump(template_estim, SAVE_DIR / "group_alignment.pkl")
    joblib.dump(pairwise_estim, SAVE_DIR / "pairwise_alignment.pkl")
    np.save(SAVE_DIR / "procrustes_template.npy", procrustes_template)
    np.save(SAVE_DIR / "atlas_labels.npy", labels)
    print(f"Saved alignment estimators and atlas labels to {SAVE_DIR}")
