"""
This module contains common functions and constants used across multiple
analysis notebooks to reduce code duplication and ensure consistency.
"""

import os
import gc
import psutil
import numpy as np
import nibabel as nib
from pathlib import Path
from collections import defaultdict

# ============================================================================
# Study Configuration Constants
# ============================================================================

TASKS = [
    "nBack", "flanker", "directedForgetting", "goNogo",
    "shapeMatching", "stopSignal", "cuedTS", "spatialTS"
]

CONTRASTS = {
    "nBack": ["twoBack-oneBack", "match-mismatch", "task-baseline", "response_time"],
    "flanker": ["incongruent-congruent", "task-baseline"],
    "directedForgetting": ["neg-con", "task-baseline", "response_time"],
    "goNogo": ["nogo_success-go", "nogo_success", "task-baseline", "response_time"],
    "shapeMatching": [
        "DDD", "DDS", "DNN", "DSD", "main_vars", "SDD", "SNN", "SSS",
        "task-baseline", "response_time"
    ],
    "stopSignal": [
        "go", "stop_failure-go", "stop_failure", "stop_failure-stop_success",
        "stop_success-go", "stop_success", "stop_success-stop_failure",
        "task-baseline", "response_time"
    ],
    "cuedTS": [
        "cue_switch_cost", "task_switch_cost",
        "task_switch_cue_switch-task_stay_cue_stay", "task-baseline", "response_time"
    ],
    "spatialTS": [
        "cue_switch_cost", "task_switch_cost",
        "task_switch_cue_switch-task_stay_cue_stay", "task-baseline", "response_time"
    ]
}

SUBJECTS = ['sub-s03', 'sub-s10', 'sub-s19', 'sub-s29', 'sub-s43']
SESSIONS = [
    'ses-01', 'ses-02', 'ses-03', 'ses-04', 'ses-05',
    'ses-06', 'ses-07', 'ses-08', 'ses-09', 'ses-10'
]
ENCOUNTERS = ['01', '02', '03', '04', '05']


# ============================================================================
# File Path Utilities
# ============================================================================

# this is to get the firs tleve
def build_first_level_contrast_map_path(base_dir, level, subject, session, task, contrast_name):
    """Build the file path for a contrast map.
    
    Handles special cases for subjects with different file naming conventions:
    - sub-s10: flanker task has 'run-1' in filename
    - sub-s03: all tasks have 'run-1' in filename
    
    Parameters
    ----------
    base_dir : str
        Base directory for the data
    level : str
        Processing level (e.g., 'output_lev1_mni')
    subject : str
        Subject ID (e.g., 'sub-s03')
    session : str
        Session ID (e.g., 'ses-01')
    task : str
        Task name (e.g., 'nBack')
    contrast_name : str
        Contrast name (e.g., 'twoBack-oneBack')
    
    Returns
    -------
    str
        Full path to the contrast map file
    """
    filename = (
        f'{subject}_{session}_task-{task}_contrast-{contrast_name}'
        f'_rtmodel-rt_centered_stat-effect-size.nii.gz'
    )
    
    # Special case: sub-s10 in flanker task
    if (subject == 'sub-s10' and task == 'flanker'):
        filename = (
            f'{subject}_{session}_run-1_task-{task}_contrast-{contrast_name}'
            f'_rtmodel-rt_centered_stat-effect-size.nii.gz'
        )
    # Special case: sub-s03 in all tasks
    elif (subject == 'sub-s03'):
        filename = (
            f'{subject}_{session}_run-1_task-{task}_contrast-{contrast_name}'
            f'_rtmodel-rt_centered_stat-effect-size.nii.gz'
        )
        
    return os.path.join(base_dir, level, subject, task, 'indiv_contrasts', filename)

def build_fixed_effects_path(base_dir, level, subject, task, contrast_name):
    """Build the file path for a fixed effects map.
    
    Parameters
    ----------
    base_dir : str
        Base directory for the data
    level : str
        Processing level (e.g., 'output_lev1_mni')
    subject : str
        Subject ID (e.g., 'sub-s03')
    task : str
        Task name (e.g., 'nBack')
    contrast_name : str
        Contrast name (e.g., 'twoBack-oneBack')
    
    Returns
    -------
    str
        Full path to the fixed effects map file
    """
    filename = (
        f'{subject}_task-{task}_contrast-{contrast_name}'
        f'_rtmodel-rt_centered_stat-fixed-effects.nii.gz'
    )
    return os.path.join(base_dir, level, subject, task, 'fixed_effects', filename)


# ============================================================================
# Data Validation and Cleaning
# ============================================================================

def is_valid_contrast_map(img_path):
    """Check if a contrast map has sufficient variance and no NaN values.
    
    Parameters
    ----------
    img_path : str
        Path to the NIfTI file to validate
    
    Returns
    -------
    bool
        True if the map is valid, False otherwise
    """
    try:
        img = nib.load(img_path)
        data = img.get_fdata()
        return np.std(data) > 1e-10 and not np.isnan(data).any()
    except Exception as e:
        print(f"Error validating {img_path}: {e}")
        return False


def clean_z_map_data(z_map, task, contrast_name, encounter):
    """Clean z-map data by handling NaN and infinity values.
    
    Parameters
    ----------
    z_map : nibabel.Nifti1Image
        The z-map image to clean
    task : str
        Task name (for logging)
    contrast_name : str
        Contrast name (for logging)
    encounter : int or str
        Encounter number (for logging)
    
    Returns
    -------
    nibabel.Nifti1Image
        Cleaned z-map image
    """
    data = z_map.get_fdata()
    if np.isnan(data).any() or np.isinf(data).any():
        data = np.nan_to_num(data)
        z_map = nib.Nifti1Image(data, z_map.affine, z_map.header)
        print(
            f"Warning: Fixed NaN/Inf values in {task}:{contrast_name}:"
            f"encounter-{encounter+1 if isinstance(encounter, int) else encounter}"
        )
    return z_map


# ============================================================================
# Memory Management
# ============================================================================

def cleanup_memory():
    """Clean up memory between batches.
    
    Forces garbage collection and prints current memory usage.
    Useful when processing large datasets to monitor memory consumption.
    """
    # Force garbage collection
    gc.collect()
    
    # Get memory info
    memory = psutil.virtual_memory()
    print(
        f"Memory after cleanup: {memory.percent:.1f}% used "
        f"({memory.available/(1024**3):.1f}GB available)"
    )


# ============================================================================
# Data Structure Utilities
# ============================================================================

def convert_to_regular_dict(d):
    """Convert defaultdict to regular dict recursively.
    
    Useful for serialization (e.g., when saving to pickle files).
    
    Parameters
    ----------
    d : dict or list
        Data structure that may contain defaultdicts
    
    Returns
    -------
    dict or list
        Same structure with all defaultdicts converted to regular dicts
    """
    if isinstance(d, defaultdict):
        return {k: convert_to_regular_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [convert_to_regular_dict(i) for i in d]
    else:
        return d


# ============================================================================
# Helper Functions for Requested Contrasts
# ============================================================================

def get_requested_task_contrasts():
    """Get a dictionary of all requested task-contrast combinations.
    
    Returns
    -------
    defaultdict
        Dictionary with structure: {task: [list of contrasts]}
    """
    requested_task_contrasts = defaultdict(lambda: defaultdict(list))
    for task in TASKS:
        requested_task_contrasts[task] = CONTRASTS[task]
    return requested_task_contrasts


def get_compiled_contrasts():
    """Get a flat list of all unique contrasts across all tasks.
    
    Returns
    -------
    list
        List of unique contrast names
    """
    compiled_contrasts = []
    for task in TASKS:
        for contrast in CONTRASTS[task]:
            if contrast not in compiled_contrasts:
                compiled_contrasts.append(contrast)
    return compiled_contrasts