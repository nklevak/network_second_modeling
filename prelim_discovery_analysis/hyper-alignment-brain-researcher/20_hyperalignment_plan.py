"""
=============================================================================
HYPERALIGNMENT PLAN FOR NEURAL PRACTICE EFFECTS
Network Study — 5 Discovery Subjects, 10 Sessions (5 Encounters), 8 Tasks
Updated with literature review — see citations throughout
=============================================================================

MOTIVATION: WHY PCA PER-SUBJECT IS INSUFFICIENT
------------------------------------------------
PCA applied per-subject identifies latent dimensions of parcel activation
patterns, but cross-subject comparison is fundamentally broken:
    - Parcel X in sub-s03 ≠ Parcel X in sub-s10 (different neural topographies)
    - PC1 may point in orthogonal directions across subjects despite both
      correlating with session/encounter number
    - Sign-alignment hacks (flipping PC1 by Spearman r with encounter) are
      a post-hoc bandaid, not a solution
    - No ground truth that all subjects' PC1 represents the same cognitive process

Hyperalignment addresses this by projecting each subject's data into a COMMON
representational space. After alignment, "dimension k" means the same cognitive
function for every subject, enabling meaningful group-level analysis.


KEY CONSTRAINT: SUBJECTS SEE DIFFERENT TRIALS
----------------------------------------------
Standard stimulus-locked hyperalignment (Haxby et al. 2011 Neuron) requires
subjects to see *identical stimuli at identical time points*. The same is true
of the standard Shared Response Model — SRM (Chen et al. 2015 NeurIPS), which
maximizes correlation at *matched time points* across subjects.

Your design: subjects perform the same TASKS (nBack, flanker, stopSignal, etc.)
across 5 encounters but with DIFFERENT trial items each time.

CONCLUSION: Standard SRM and standard response hyperalignment DO NOT apply.

WHAT DOES APPLY:
    (A) Connectivity-Based Hyperalignment (CHA) — no stimulus matching needed
    (B) Condition/Contrast-Level Procrustes — conditions ARE matched across subjects
    (C) Hybrid Hyperalignment — combines (A) and response alignment
    (D) Connectivity-SRM (cSRM) — SRM variant on connectivity fingerprints


=============================================================================
RECOMMENDED METHODS (ranked by fit to your design)
=============================================================================

─────────────────────────────────────────────────────────────────────────────
METHOD A (RECOMMENDED): Condition/Contrast-Level Procrustes Hyperalignment
─────────────────────────────────────────────────────────────────────────────
Reference: Supervised/task-based hyperalignment literature
           Haxby et al. 2011 Neuron (Procrustes math)
           Nastase et al. 2020 NeuroImage (task-based validation)

WHY IT FITS:
    Your subjects may see different trial items, but they ALL experience the
    same *task conditions* (contrast labels) — e.g., twoBack-oneBack,
    incongruent-congruent, stop_success-go.

    The key insight: you can build a (N_contrasts × 429_parcels) matrix per
    subject where ROWS are semantically matched across subjects (same contrast
    = same cognitive demand), even if the underlying trials differ.

    Procrustes on this matrix finds an orthogonal rotation R_i for each subject
    such that their contrast geometry aligns to a shared template. This is
    principled because the condition structure is preserved.

HOW TO BUILD THE ALIGNMENT MATRIX:
    Option 1 — Pool contrasts across all sessions (most data, most stable):
        For each subject, stack all (task × contrast × session) beta maps
        → matrix of shape (N_contrasts_total × 429), where N_contrasts_total
          = sum across tasks of (n_contrasts × n_encounters)
        Rows are: nBack/twoBack-oneBack/enc1, nBack/twoBack-oneBack/enc2, ...,
                  flanker/incongruent-congruent/enc1, ...
        Subjects must have matching rows — handle missing encounters carefully.

    Option 2 — Use session-averaged contrast maps (simpler, less noisy):
        For each subject, average beta values across available encounters
        → matrix of shape (N_contrasts_unique × 429)
        This is more stable with N=5 subjects and 5 encounters.
        Recommended for estimating the alignment transformation.

    Option 3 — Per-session alignment (most ambitious):
        Run Procrustes separately per session → track how alignment quality
        changes with practice. Requires more data per fit; risky with N=5.
        Prefer Option 2 for stable transformations, then apply to session data.

IMPLEMENTATION:
    from scipy.linalg import orthogonal_procrustes
    import numpy as np

    def build_contrast_matrix(parcel_dict, subjects, tasks, contrasts_by_task,
                               encounters=None, agg='mean'):
        '''
        Build (N_contrasts × 429) matrix per subject.
        agg: 'mean' averages across encounters; 'stack' stacks all encounter rows.
        Returns dict: {subject -> np.ndarray (N_rows × 429)}
        '''
        matrices = {}
        for subj in subjects:
            rows = []
            for task in tasks:
                for contrast in contrasts_by_task[task]:
                    enc_data = []
                    for enc in (encounters or ['01','02','03','04','05']):
                        try:
                            df = parcel_dict[subj][task][contrast][enc]
                            enc_data.append(df.set_index('region')['activation'].values)
                        except KeyError:
                            pass
                    if enc_data:
                        if agg == 'mean':
                            rows.append(np.mean(enc_data, axis=0))
                        else:
                            rows.extend(enc_data)
            if rows:
                matrices[subj] = np.array(rows)  # (N_rows × 429)
        return matrices


    def iterative_procrustes(matrices, n_iter=20):
        '''
        Iterative group Procrustes hyperalignment.
        Finds rotation R_i for each subject minimizing ||X_i @ R_i - T|| where
        T is the iteratively updated group template.

        Returns: rotations dict, aligned_matrices dict, template
        '''
        subjects = list(matrices.keys())
        # Initialize template as mean of all subjects
        template = np.mean(np.stack(list(matrices.values())), axis=0)

        for iteration in range(n_iter):
            rotations = {}
            aligned = {}
            for subj in subjects:
                X = matrices[subj]
                # orthogonal_procrustes finds R minimizing ||X @ R - T||_F
                R, _ = orthogonal_procrustes(X, template)
                rotations[subj] = R
                aligned[subj] = X @ R

            # Update template (mean of aligned data)
            new_template = np.mean(np.stack(list(aligned.values())), axis=0)
            change = np.linalg.norm(new_template - template)
            template = new_template
            print(f"Iter {iteration+1}: template change = {change:.6f}")
            if change < 1e-6:
                print("Converged.")
                break

        return rotations, aligned, template


─────────────────────────────────────────────────────────────────────────────
METHOD B: Connectivity-Based Hyperalignment (CHA)
─────────────────────────────────────────────────────────────────────────────
Reference: Nastase et al. (2020) NeuroImage
           "Leveraging shared connectivity to aggregate heterogeneous datasets
            into a common response space"
           GitHub: https://github.com/snastase/connectivity-srm

WHY IT FITS:
    CHA uses each parcel's functional connectivity PROFILE (its pattern of
    correlations with all other parcels) as the alignment signal. Because
    connectivity profiles are "second-order" — they describe the geometry of
    the representational space rather than specific activation values — they
    are homologous across subjects even without identical stimuli.

    Nastase et al. explicitly designed CHA for heterogeneous datasets with
    no shared stimulus paradigm.

WHAT YOU NEED:
    Parcellated timeseries data (not just betas). If you have the preprocessed
    BOLD timeseries parcellated into 429 regions from each session, you can:
    1. Estimate a (429 × 429) connectivity matrix per subject per session
    2. Use the connectivity geometry to find Procrustes/SRM transformations
    3. Apply those transformations to the beta/effect-size maps

    If you only have beta maps (no timeseries), skip to Method A.

IMPLEMENTATION SKETCH:
    # Connectivity-SRM using BrainIAK + connectivity fingerprints
    # See Nastase et al. 2020 for full code
    from brainiak.funcalign.srm import SRM
    import numpy as np

    def build_connectivity_fingerprints(timeseries_list, n_targets=429):
        '''
        For each subject, compute (n_targets × n_timepoints) connectivity
        profile matrix (correlation of each parcel with all other parcels).

        timeseries_list: list of (n_parcels × n_timepoints) per subject
        Returns: list of (n_parcels × n_parcels) connectivity matrices
        '''
        fingerprints = []
        for ts in timeseries_list:
            # Correlation matrix (429 × 429)
            corr = np.corrcoef(ts)  # or partial correlation
            fingerprints.append(corr)
        return fingerprints

    # Then run SRM on the connectivity fingerprints (n_parcels × n_parcels input)
    # SRM finds shared components of the connectivity geometry
    srm = SRM(n_iter=50, features=20)
    srm.fit(fingerprints)   # srm.w_: 429 × k shared basis


─────────────────────────────────────────────────────────────────────────────
METHOD C: Hybrid Hyperalignment
─────────────────────────────────────────────────────────────────────────────
Reference: Feilong et al. (2021) NeuroImage
           "Hybrid hyperalignment: A single high-dimensional model of shared
            information embedded in cortical patterns of response and
            functional connectivity"
           PubMed: 33762217

WHY IT FITS:
    Combines connectivity-based AND response-based alignment information in
    a single objective. Best performance on both fronts, but requires both
    timeseries AND task data. Use if you have parcellated BOLD timeseries.


=============================================================================
RECOMMENDED PIPELINE FOR YOUR SPECIFIC DATA
=============================================================================

Given that you have beta/effect-size maps (not raw timeseries), and subjects
performed the same task conditions with different trial items:

STEP 1: Build the alignment matrix (Method A)
    → For each subject, average beta maps across encounters per contrast
    → Result: (N_contrasts × 429) matrix per subject, semantically matched
    → N_contrasts = sum over tasks of n_contrasts per task
      (roughly 8 tasks × ~4 contrasts avg = ~32 rows per subject)

STEP 2: Estimate alignment transformations
    → Run iterative Procrustes (iterative_procrustes above)
    → Get rotation matrix R_i (429 × 429) per subject

STEP 3: Apply to session-level data
    → For each subject/task/contrast/encounter:
          aligned_beta = raw_beta @ R_i
    → Now all subjects' data lives in the same parcel space

STEP 4: Practice effect analysis in common space (see analyses below)

STEP 5: Validate alignment quality
    → Cross-subject decoding: train on N-1 subjects, predict task condition
      in the held-out subject. Alignment should improve this accuracy.
    → Compare ISC (inter-subject correlation) before vs. after alignment.


=============================================================================
ANALYSES FOR INVESTIGATING PRACTICE EFFECTS IN COMMON SPACE
=============================================================================

After alignment, you have per-session beta maps in a common parcel space for
all 5 subjects. These analyses address neural practice effects:

─────────────────────────────────────────────────────────────────────────────
ANALYSIS 1 (PRIMARY): Linear Mixed-Effects Model per Parcel
─────────────────────────────────────────────────────────────────────────────
Reference: Chen et al. (2013) NeuroImage PMCID: PMC3638840

    For each parcel in aligned space:
        activation ~ encounter_centered + (1 + encounter_centered | subject)

    Fixed-effect slope = group-level practice effect (increase or decrease)
    Random slope = individual rate of change
    Multiple comparison correction: FDR or permutation across 429 parcels

    This is strictly better than your current individual slope approach
    because subjects are now in a common space — averaging slopes across
    subjects is meaningful.

    Implementation:
        import statsmodels.formula.api as smf
        import pandas as pd

        def run_lme_practice_effects(aligned_parcel_dict, subjects, tasks,
                                      contrasts_by_task):
            results = {}
            for task in tasks:
                results[task] = {}
                for contrast in contrasts_by_task[task]:
                    rows = []
                    for subj in subjects:
                        for enc_idx, enc in enumerate(['01','02','03','04','05']):
                            try:
                                beta_aligned = aligned_parcel_dict[subj][task][contrast][enc]
                                # beta_aligned: 429-dim vector in common space
                                for parcel_idx, val in enumerate(beta_aligned):
                                    rows.append({
                                        'subject': subj,
                                        'encounter': enc_idx + 1,
                                        'encounter_c': enc_idx + 1 - 3.0,
                                        'parcel': parcel_idx,
                                        'activation': val
                                    })
                            except KeyError:
                                pass
                    df = pd.DataFrame(rows)
                    parcel_results = {}
                    for parcel in df['parcel'].unique():
                        pdf = df[df['parcel'] == parcel]
                        try:
                            model = smf.mixedlm(
                                "activation ~ encounter_c",
                                pdf, groups=pdf["subject"],
                                re_formula="~encounter_c"
                            ).fit(reml=False)
                            parcel_results[parcel] = {
                                'slope': model.fe_params['encounter_c'],
                                'p_value': model.pvalues['encounter_c'],
                                'converged': model.converged
                            }
                        except Exception as e:
                            parcel_results[parcel] = {'slope': np.nan, 'error': str(e)}
                    results[task][contrast] = parcel_results
            return results


─────────────────────────────────────────────────────────────────────────────
ANALYSIS 2: Inter-Subject Correlation (ISC) Across Encounters
─────────────────────────────────────────────────────────────────────────────
In the aligned space, for each encounter compute pairwise Pearson correlation
between subjects' 429-parcel activation vectors. Plot mean pairwise ISC as a
function of encounter number.

Hypothesis: if practice makes neural patterns more stereotyped/efficient,
ISC should INCREASE across encounters (subjects converge).
Conversely, decreasing ISC suggests neural individuation with practice.

This is the clearest direct test of convergence/divergence.

    def compute_isc_across_encounters(aligned_parcel_dict, subjects, task, contrast):
        from itertools import combinations
        encounters = ['01', '02', '03', '04', '05']
        isc_by_enc = {}
        for enc in encounters:
            vecs = []
            for subj in subjects:
                try:
                    vecs.append(aligned_parcel_dict[subj][task][contrast][enc])
                except KeyError:
                    pass
            if len(vecs) >= 2:
                pairs = list(combinations(range(len(vecs)), 2))
                pairwise_r = [np.corrcoef(vecs[i], vecs[j])[0,1] for i,j in pairs]
                isc_by_enc[enc] = np.mean(pairwise_r)
        return isc_by_enc  # {encounter -> mean pairwise ISC}


─────────────────────────────────────────────────────────────────────────────
ANALYSIS 3: Representational Similarity Analysis (RSA) Across Encounters
─────────────────────────────────────────────────────────────────────────────
Reference: Kriegeskorte et al. (2008) Front. Systems Neuroscience PMC2605405
           Freund et al. (2021) Front. Neurology PMC8666569

For each subject × encounter, build a (N_contrasts × N_contrasts) RDM using
the 429-parcel activation vectors as representations of each contrast.

Then ask:
    (a) Does the RDM structure change across encounters? (practice reshapes
        the representational geometry)
    (b) Does the RDM converge toward the group template across encounters?
    (c) Are RDMs more similar to a "task difficulty" model RDM (effort model)
        in early encounters, and more similar to a "cognitive demand" model
        in later encounters?

This tests whether practice changes not just the LEVEL of activation but the
STRUCTURE of how different conditions are represented.

    def compute_rdm(parcel_dict, subject, task, contrasts, encounter):
        '''Compute (N_contrasts × N_contrasts) RDM for one subject/encounter.'''
        from scipy.spatial.distance import pdist, squareform
        vecs = []
        labels = []
        for contrast in contrasts:
            try:
                v = parcel_dict[subject][task][contrast][encounter]
                vecs.append(v)
                labels.append(contrast)
            except KeyError:
                pass
        if len(vecs) < 2:
            return None, None
        dist_matrix = squareform(pdist(np.array(vecs), metric='correlation'))
        return dist_matrix, labels

    def rdm_trajectory_analysis(parcel_dict, subjects, task, contrasts):
        '''
        Compute per-subject, per-encounter RDMs and measure how much they
        change across encounters (using Spearman r between RDMs).
        '''
        from scipy.stats import spearmanr
        encounters = ['01', '02', '03', '04', '05']
        for subj in subjects:
            rdms = {}
            for enc in encounters:
                rdm, labels = compute_rdm(parcel_dict, subj, task, contrasts, enc)
                if rdm is not None:
                    rdms[enc] = rdm
            # Correlate encounter 1 RDM with all subsequent encounters
            enc_list = sorted(rdms.keys())
            if enc_list:
                ref = rdms[enc_list[0]].flatten()
                for enc in enc_list[1:]:
                    r, p = spearmanr(ref, rdms[enc].flatten())
                    print(f"  {subj} enc01 vs {enc}: r={r:.3f}, p={p:.3f}")


─────────────────────────────────────────────────────────────────────────────
ANALYSIS 4: IS-RSA — Link Neural Change to Behavioral Change
─────────────────────────────────────────────────────────────────────────────
Reference: Finn et al. (2023) PNAS doi:10.1073/pnas.2308951120

Inter-Subject RSA tests: do subjects with similar behavioral learning curves
also show similar neural learning curves?

    1. Build a 5×5 behavioral similarity matrix from RT/accuracy improvement slopes
    2. Build a 5×5 neural similarity matrix (correlation of practice slope maps
       across subjects in common space)
    3. Correlate the two matrices (Mantel test)

    This links individual differences in learning speed to individual differences
    in neural change — a key question for cognitive training research.


─────────────────────────────────────────────────────────────────────────────
ANALYSIS 5: Cross-Subject Decoding (Alignment Quality Validation)
─────────────────────────────────────────────────────────────────────────────
    1. Before alignment: train SVM on contrast maps from subjects 1-4,
       predict task condition for subject 5. Repeat leave-one-out.
    2. After alignment: same procedure.
    3. Alignment should improve cross-subject generalization.
    4. BONUS: run decoding separately per encounter. Improvement in cross-subject
       decoding accuracy across encounters = practice makes representations
       more consistent across individuals.

    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score

    def cross_subject_decoding(aligned_parcel_dict, subjects, task, contrasts):
        encounters = ['01', '02', '03', '04', '05']
        results_by_enc = {}
        for enc in encounters:
            X_all, y_all, groups = [], [], []
            for subj in subjects:
                for contrast in contrasts:
                    try:
                        vec = aligned_parcel_dict[subj][task][contrast][enc]
                        X_all.append(vec)
                        y_all.append(contrast)
                        groups.append(subj)
                    except KeyError:
                        pass
            # Leave-one-subject-out cross-validation
            from sklearn.model_selection import LeaveOneGroupOut
            logo = LeaveOneGroupOut()
            clf = SVC(kernel='linear', C=1.0)
            scores = cross_val_score(clf, X_all, y_all, groups=groups, cv=logo)
            results_by_enc[enc] = scores.mean()
            print(f"Encounter {enc}: cross-subject accuracy = {scores.mean():.3f}")
        return results_by_enc


=============================================================================
CRITICAL DESIGN DECISIONS
=============================================================================

1. WHAT TO USE AS ALIGNMENT TARGET:
   Session-averaged betas (averaged across encounters) give the most stable
   estimate of the transformation, because practice effects are small relative
   to condition effects. Align to the "average brain state" then study how
   sessions deviate from this average.

2. REGULARIZATION (N=5 SUBJECTS):
   Full Procrustes (429×429 rotation) has many parameters relative to N=5.
   Options:
   (a) Low-rank Procrustes: truncate SVD to k components (k~20-50)
   (b) Ridge-regularized alignment (fmralign package)
   (c) Parcel-subset: align only using parcels with high inter-subject
       variability (top 50% by std across subjects)

3. ENCOUNTER-BY-ENCOUNTER vs. SINGLE TRANSFORMATION:
   Preferred: estimate ONE transformation per subject (from session-averaged
   data) and apply it to ALL encounters. This lets you study how the
   same coordinate system evolves with practice, without confounding practice
   effects with alignment instability.

4. MISSING ENCOUNTERS:
   Use only present encounters in the alignment matrix. Do NOT impute.
   For LME analysis, missingness at random is handled naturally by the LME.


=============================================================================
PACKAGE REQUIREMENTS
=============================================================================

# Core (likely already installed)
pip install scipy numpy pandas statsmodels scikit-learn

# For connectivity hyperalignment (optional, if you have timeseries)
pip install brainiak          # SRM, CHA
pip install fmralign          # Procrustes, Ridge, ScaledOrthogonal alignment

# Visualization
pip install nilearn matplotlib seaborn


=============================================================================
REFERENCES
=============================================================================

1. Haxby JV et al. (2011). A common, high-dimensional model of the
   representational space in human ventral temporal cortex. Neuron, 72, 404-416.
   → Original response hyperalignment (requires identical stimuli)

2. Chen PH et al. (2015). A reduced-dimension fMRI shared response model.
   NeurIPS. → Standard SRM (ALSO requires temporal synchrony — not for you)

3. Nastase SA et al. (2020). Leveraging shared connectivity to aggregate
   heterogeneous datasets into a common response space. NeuroImage, 116865.
   PMC7958465. GitHub: https://github.com/snastase/connectivity-srm
   → Connectivity hyperalignment — works WITHOUT identical stimuli

4. Feilong M et al. (2021). Hybrid hyperalignment: A single high-dimensional
   model of shared information embedded in cortical patterns of response and
   functional connectivity. NeuroImage, 117975. PubMed: 33762217.
   → Best method if you have both timeseries and task data

5. Haxby JV et al. (2020). Hyperalignment: Modeling shared information encoded
   in idiosyncratic cortical topographies. eLife, 56601.
   → Review; confirms resting-state CHA works for novel-stimulus generalization

6. Chen G et al. (2013). Linear mixed-effects modeling approach to fMRI group
   analysis. NeuroImage, 73, 176-190. PMC3638840.
   → LME as primary analysis for group-level practice effects

7. Kriegeskorte N et al. (2008). Representational similarity analysis.
   Front. Systems Neuroscience. PMC2605405. → RSA methodology

8. Finn ES et al. (2023). Intersubject similarity in neural representations
   underlies shared episodic memory content. PNAS. doi:10.1073/pnas.2308951120
   → IS-RSA for linking individual differences in neural change to behavior

9. Freund MC et al. (2021). Multivariate fMRI signatures of learning in a
   Hebb repetition paradigm. Front. Neurology. PMC8666569.
   → RSA applied to tracking representational change across sessions
"""

# =============================================================================
# QUICK-START: verify your data is ready for alignment
# =============================================================================

import numpy as np
import pickle
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr
import warnings

# --- CONFIG (adjust paths as needed) ---
DATA_DIR = Path("/oak/stanford/groups/russpold/users/nklevak/network_second_modeling/processed_data_dfs_2026")
SUBJECTS = ["sub-s03", "sub-s10", "sub-s19", "sub-s29", "sub-s43"]
ENCOUNTERS = ["01", "02", "03", "04", "05"]
TASKS = ["nBack", "flanker", "directedForgetting", "goNogo",
         "shapeMatching", "stopSignal", "cuedTS", "spatialTS"]
CONTRASTS_BY_TASK = {
    "nBack":              ["twoBack-oneBack", "match-mismatch", "task-baseline"],
    "flanker":            ["incongruent-congruent", "task-baseline"],
    "directedForgetting": ["neg-con", "task-baseline"],
    "goNogo":             ["nogo_success-go", "nogo_success", "task-baseline"],
    "shapeMatching":      ["main_vars", "task-baseline"],
    "stopSignal":         ["stop_success-go", "stop_failure-go", "task-baseline"],
    "cuedTS":             ["cue_switch_cost", "task_switch_cost", "task-baseline"],
    "spatialTS":          ["cue_switch_cost", "task_switch_cost", "task-baseline"],
}
N_PARCELS = 429


def load_parcel_dict(data_dir=DATA_DIR):
    """Load parcellated beta data."""
    pkl_files = sorted((data_dir / "smor_parcel_dfs").glob(
        "discovery_parcel_indiv_mean_updated_0111_*_betas.pkl"
    ))
    if not pkl_files:
        raise FileNotFoundError(f"No parcel pkl files found in {data_dir}/smor_parcel_dfs/")
    # Load the first file; adjust if data is split across multiple files
    with open(pkl_files[0], "rb") as f:
        return pickle.load(f)


def build_contrast_matrix(parcel_dict, subject, tasks=TASKS,
                           contrasts_by_task=CONTRASTS_BY_TASK,
                           encounters=ENCOUNTERS, agg="mean"):
    """
    Build (N_contrasts × N_PARCELS) matrix for one subject.
    agg='mean': average betas across available encounters (recommended for alignment).
    agg='stack': stack all encounter rows (more data but assumes encounter ordering).

    Returns: np.ndarray (N_rows × N_PARCELS), list of row labels
    """
    rows, labels = [], []
    for task in tasks:
        for contrast in contrasts_by_task.get(task, []):
            enc_data = []
            for enc in encounters:
                try:
                    df = parcel_dict[subject][task][contrast][enc]
                    v = df.set_index("region")["activation"].values
                    if len(v) == N_PARCELS:
                        enc_data.append(v)
                except KeyError:
                    pass
            if enc_data:
                if agg == "mean":
                    rows.append(np.mean(enc_data, axis=0))
                else:  # stack
                    rows.extend(enc_data)
                labels.append(f"{task}__{contrast}")
    return np.array(rows), labels  # (N_rows × 429), list of labels


def iterative_procrustes(matrices_dict, n_iter=20, tol=1e-6):
    """
    Iterative group Procrustes hyperalignment.

    matrices_dict: {subject -> np.ndarray (N_rows × N_PARCELS)}
                   All matrices must have the same number of rows.

    Returns:
        rotations:  {subject -> R (N_PARCELS × N_PARCELS)}
        aligned:    {subject -> aligned matrix (N_rows × N_PARCELS)}
        template:   group template (N_rows × N_PARCELS)
    """
    subjects = list(matrices_dict.keys())
    template = np.mean(np.stack(list(matrices_dict.values())), axis=0)

    for it in range(n_iter):
        rotations, aligned = {}, {}
        for subj in subjects:
            X = matrices_dict[subj]
            R, _ = orthogonal_procrustes(X, template)
            rotations[subj] = R
            aligned[subj] = X @ R
        new_template = np.mean(np.stack(list(aligned.values())), axis=0)
        change = np.linalg.norm(new_template - template)
        template = new_template
        if change < tol:
            print(f"Procrustes converged at iteration {it+1} (change={change:.2e})")
            break
    else:
        print(f"Procrustes reached max iterations (final change={change:.2e})")

    return rotations, aligned, template


def apply_rotation_to_session_data(parcel_dict, subject, rotation_R,
                                    tasks=TASKS, contrasts_by_task=CONTRASTS_BY_TASK,
                                    encounters=ENCOUNTERS):
    """
    Apply the Procrustes rotation R to all session-level beta maps for one subject.

    Returns: {task -> {contrast -> {encounter -> aligned_vector (N_PARCELS,)}}}
    """
    aligned_dict = {}
    for task in tasks:
        aligned_dict[task] = {}
        for contrast in contrasts_by_task.get(task, []):
            aligned_dict[task][contrast] = {}
            for enc in encounters:
                try:
                    df = parcel_dict[subject][task][contrast][enc]
                    v = df.set_index("region")["activation"].values
                    if len(v) == N_PARCELS:
                        aligned_dict[task][contrast][enc] = v @ rotation_R
                except KeyError:
                    pass
    return aligned_dict


def compute_alignment_quality(before_dict, after_dict, subjects=SUBJECTS):
    """
    Compute mean pairwise inter-subject correlation before and after alignment.
    Higher ISC after alignment = better.
    """
    from itertools import combinations

    def mean_isc(data_dict):
        vecs = [np.concatenate([v for task in data_dict[s].values()
                                 for c_data in task.values()
                                 for v in c_data.values()])
                for s in subjects if s in data_dict]
        if len(vecs) < 2:
            return np.nan
        return np.mean([pearsonr(vecs[i], vecs[j])[0]
                        for i, j in combinations(range(len(vecs)), 2)])

    isc_before = mean_isc(before_dict)
    isc_after = mean_isc(after_dict)
    print(f"Mean ISC before alignment: {isc_before:.4f}")
    print(f"Mean ISC after  alignment: {isc_after:.4f}")
    print(f"Improvement: {isc_after - isc_before:+.4f}")
    return isc_before, isc_after


# =============================================================================
# MAIN PIPELINE DEMO
# =============================================================================

if __name__ == "__main__":
    print("Loading parcel data...")
    parcel_dict = load_parcel_dict()
    print(f"Loaded. Subjects: {list(parcel_dict.keys())}")

    # --- Step 1: Build session-averaged contrast matrices ---
    print("\nBuilding contrast matrices (session-averaged)...")
    contrast_matrices = {}
    for subj in SUBJECTS:
        mat, labels = build_contrast_matrix(parcel_dict, subj, agg="mean")
        contrast_matrices[subj] = mat
        print(f"  {subj}: {mat.shape} (contrasts × parcels)")

    # Verify all subjects have the same number of rows
    n_rows = [v.shape[0] for v in contrast_matrices.values()]
    if len(set(n_rows)) > 1:
        warnings.warn(f"Subjects have different contrast counts: {dict(zip(SUBJECTS, n_rows))}. "
                       "Handle missing contrasts before proceeding.")

    # --- Step 2: Iterative Procrustes hyperalignment ---
    print("\nRunning iterative Procrustes hyperalignment...")
    rotations, aligned_avg, template = iterative_procrustes(contrast_matrices)

    # --- Step 3: Apply rotations to all session data ---
    print("\nApplying rotations to session-level data...")
    aligned_session_data = {}
    for subj in SUBJECTS:
        aligned_session_data[subj] = apply_rotation_to_session_data(
            parcel_dict, subj, rotations[subj]
        )

    # --- Step 4: Compute alignment quality ---
    print("\nAlignment quality:")
    compute_alignment_quality(parcel_dict, aligned_session_data)

    print("\nDone. Next steps:")
    print("  1. Run LME on aligned_session_data (Analysis 1 above)")
    print("  2. Compute ISC per encounter (Analysis 2)")
    print("  3. Run RSA across encounters (Analysis 3)")
    print("  4. Run cross-subject decoding to validate (Analysis 5)")
