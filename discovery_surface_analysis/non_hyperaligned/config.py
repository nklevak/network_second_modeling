import sys
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SURFACE_BASE_DIR = Path('/scratch/groups/russpold/logben/discovery_bids/derivatives/lev1_surface')
ANALYSIS_DIR     = Path(__file__).resolve().parent
DATA_DIR         = ANALYSIS_DIR / "processed_data"
FIGURES_DIR      = ANALYSIS_DIR / "figures"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Study constants ───────────────────────────────────────────────────────────
sys.path.insert(0, str(ANALYSIS_DIR.parent.parent / "prelim_discovery_analysis"))
from utils import TASKS, CONTRASTS, SUBJECTS, SESSIONS

RT_CONTRAST         = "response_time"
N_VERTICES_PER_HEMI = 40962   # fsaverage6
N_PARCELS           = 400     # Schaefer 400
N_ENCOUNTERS        = 5
ENCOUNTER_CENTER    = 3.0     # fixed center for linreg (mean of [1..5])


# ── Path builder ──────────────────────────────────────────────────────────────
def build_surface_contrast_map_path(subject, session, task, contrast, hemi, file_type="default"):
    file_ending = "z_score" if file_type == "z" else "effect-size"
    filename = (
        f'{subject}_{session}_task-{task}_run-1_hemi-{hemi}'
        f'_contrast-{contrast}_rtmodel-RTDur_stat-{file_ending}.func.gii'
    )
    return SURFACE_BASE_DIR / subject / f'task-{task}' / 'indiv_contrasts' / filename


# ── Schaefer 400 surface labels (cached) ──────────────────────────────────────
_LABELS_PATH = DATA_DIR / "schaefer400_surface_labels.npy"
_NAMES_PATH  = DATA_DIR / "schaefer400_parcel_names.npy"

def get_schaefer_surface_labels():
    """
    Return (labels, parcel_names) for Schaefer 400 on fsaverage6.
      labels       : np.ndarray (81924,) int — parcel IDs 1-400, medial wall = 401
      parcel_names : list of 400 strings, e.g. '7Networks_LH_Vis_1'
    Computed once and cached in DATA_DIR.
    """
    if _LABELS_PATH.exists() and _NAMES_PATH.exists():
        labels       = np.load(_LABELS_PATH)
        parcel_names = np.load(_NAMES_PATH, allow_pickle=True).tolist()
        return labels, parcel_names

    from nilearn.datasets import fetch_surf_fsaverage, fetch_atlas_schaefer_2018
    from nilearn.surface import vol_to_surf

    print("Computing Schaefer 400 surface labels for fsaverage6 (one-time setup)...")
    schaefer = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=1)
    fsavg6   = fetch_surf_fsaverage('fsaverage6')

    lh = vol_to_surf(schaefer.maps, fsavg6.pial_left,  interpolation='nearest_most_frequent', radius=0).astype(int)
    rh = vol_to_surf(schaefer.maps, fsavg6.pial_right, interpolation='nearest_most_frequent', radius=0).astype(int)
    labels = np.concatenate([lh, rh])
    labels[labels == 0] = 401   # medial wall → catch-all

    parcel_names = [n.decode() if isinstance(n, bytes) else n for n in schaefer.labels]

    np.save(_LABELS_PATH, labels)
    np.save(_NAMES_PATH,  np.array(parcel_names, dtype=object))
    print(f"  Cached to {DATA_DIR}")
    return labels, parcel_names


# ── Network parsing ───────────────────────────────────────────────────────────
_NETWORK_MAP = {
    'Vis': 'Visual', 'SomMot': 'Somatomotor', 'DorsAttn': 'DorsalAttention',
    'SalVentAttn': 'SalVentAttn', 'Limbic': 'Limbic', 'Cont': 'Control',
    'Default': 'Default',
}

def parse_network(parcel_name):
    """Extract network label from e.g. '7Networks_LH_Vis_1' → 'Visual'."""
    parts = parcel_name.strip().split('_')
    if len(parts) < 3:
        return 'Unknown'
    return _NETWORK_MAP.get(parts[2], parts[2])
