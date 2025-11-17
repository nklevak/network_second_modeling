"""
This module centralizes all configuration settings including data paths,
output directories, and analysis parameters. Paths can be overridden
using environment variables for different computing environments.
"""

import os
from pathlib import Path

# ============================================================================
# Data Paths
# ============================================================================

# Base directory for input data
# Can be overridden with NETWORK_DATA_DIR environment variable
BASE_DIR = '/oak/stanford/groups/russpold/data/network_grant/discovery_BIDS_20250402/derivatives/'

# Processing level directory
INPUT_LEVEL = 'output_lev1_mni'

# Behavioral data directory (if different from main data directory)
BEHAVIORAL_BASE_DIR = '/oak/stanford/groups/russpold/data/network_grant/behavioral_data/qc/discovery/'

# ============================================================================
# Output Directories
# ============================================================================

# Relative to prelim_discovery_analysis directory
OUTPUT_DIRS = {
    'schafer': 'schafer400_dfs',
    'smor': 'smor_parcel_dfs',
    'smor_fixed': 'smor_parcel_dfs_fixed',
    'correlations': 'correlation_analysis_results',
    'neurosynth': 'neurosynth_rois',
    'second_level_models': '/home/users/nklevak/network_data_second_lev/'
}

# ============================================================================
# Atlas Paths
# ============================================================================

# Paths to atlas files (relative to prelim_discovery_analysis directory)
ATLAS_PATHS = {
    'schafer': None,  # Loaded via nilearn.datasets.fetch_atlas_schaefer_2018()
    'smor': 'smor_parcel_dfs/smorgasbord_atlas_files/smorgasbord_atlas.pkl',
}

# ============================================================================
# File Naming Conventions
# ============================================================================

# Date strings used in output filenames
# Update these when creating new output files
OUTPUT_DATES = {
    'schafer': '1001',
    'smor': '1027',
    'smor_fixed': '1116',
}

# File naming patterns
FILE_PATTERNS = {
    'parcel_indiv_mean': 'discovery_parcel_indiv_mean_updated_{date}_{run_num}.pkl',
    'parcel_indiv_mean_averaged': 'discovery_parcel_indiv_mean_updated_{date}_averaged.pkl',
    'parcel_fixedeffects_mean': 'discovery_parcel_fixedeffects_mean_updated_{date}_{run_num}.pkl',
    'parcel_fixedeffects_indiv': 'discovery_parcel_fixedeffects_indiv_updated_{date}_{run_num}.pkl',
}

# ============================================================================
# Analysis Parameters
# ============================================================================

# Number of ROIs for Schaefer atlas
SCHAEFER_N_ROIS = 400

# Parcellation strategy for NiftiLabelsMasker
# Options: 'mean', 'median', 'sum', 'minimum', 'maximum', 'standard_deviation'
PARCELLATION_STRATEGY = 'mean'

# Memory cache directory for nilearn
NILEARN_CACHE_DIR = 'nilearn_cache'

# ============================================================================
# Helper Functions
# ============================================================================

def get_output_path(output_type, filename=None, date=None, run_num=None):
    """Get the full path to an output directory or file.
    
    Parameters
    ----------
    output_type : str
        Key from OUTPUT_DIRS (e.g., 'schafer', 'smor')
    filename : str, optional
        If provided, returns path to file within the output directory.
        If filename matches a FILE_PATTERNS key, will format it with date/run_num.
    date : str, optional
        Date string to format into the filename pattern.
    run_num : int or str, optional
        Run number to format into the filename pattern.
    
    Returns
    -------
    Path
        Path object to the output directory or file
    """
    if output_type not in OUTPUT_DIRS:
        raise ValueError(f"Unknown output type: {output_type}. "
                        f"Must be one of: {list(OUTPUT_DIRS.keys())}")
    
    # Get the directory containing this config file
    config_dir = Path(__file__).parent
    output_dir = config_dir / OUTPUT_DIRS[output_type]
    
    if filename:
        # If filename matches a pattern key, format it
        if filename in FILE_PATTERNS:
            if not date:
                date = OUTPUT_DATES.get(output_type, 'unknown_date')
            formatted_filename = FILE_PATTERNS[filename].format(date=date, run_num=run_num)
            return output_dir / formatted_filename
        return output_dir / filename
    return output_dir


def get_atlas_path(atlas_name):
    """Get the full path to an atlas file.
    
    Parameters
    ----------
    atlas_name : str
        Key from ATLAS_PATHS (e.g., 'schafer', 'smor')
    
    Returns
    -------
    Path or None
        Path object to the atlas file, or None if loaded via nilearn
    """
    if atlas_name not in ATLAS_PATHS:
        raise ValueError(f"Unknown atlas: {atlas_name}. "
                        f"Must be one of: {list(ATLAS_PATHS.keys())}")
    
    if ATLAS_PATHS[atlas_name] is None:
        return None
    
    # Get the directory containing this config file
    config_dir = Path(__file__).parent
    return config_dir / ATLAS_PATHS[atlas_name]


def ensure_output_dir(output_type):
    """Ensure an output directory exists, creating it if necessary.
    
    Parameters
    ----------
    output_type : str
        Key from OUTPUT_DIRS (e.g., 'schafer', 'smor')
    
    Returns
    -------
    Path
        Path object to the output directory
    """
    output_path = get_output_path(output_type)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path