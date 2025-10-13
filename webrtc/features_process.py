import glob
import pdb
import json
import numpy as np
from sklearn.cluster import DBSCAN
import re
import pandas as pd
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import shutil
from matplotlib.patches import Patch
from collections import Counter
import heapq
import math
import itertools
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from scipy.stats import spearmanr,pearsonr,t as tdist, norm
import statsmodels.api as sm
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
from collections import defaultdict, OrderedDict
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance
import textwrap
from itertools import chain, combinations
from matplotlib.patches import Rectangle
from typing import Dict, Any, Iterable, Tuple, List, Optional, Hashable, Set

from statsmodels.formula.api import ols
import statsmodels.stats.multicomp as mc
import warnings
from statsmodels.stats.multitest import multipletests
import patsy
from scipy.stats import kruskal,friedmanchisquare
import scikit_posthocs as sp 
from scipy.stats import fisher_exact, ttest_ind,mannwhitneyu
from scipy.stats import wilcoxon, rankdata
import scipy.stats as st
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

pd.set_option('display.max_rows', None)


nasa_questions = ['Mental Demand: How mentally demanding was the task?', 'Physical Demand: How physically demanding was the task?', 'Temporal Demand: How hurried or rushed was the pace of the task?', 'Performance: How successful were you in accomplishing what you were asked to do?', 'Effort: How hard did you have to work to accomplish your level of performance?', 'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?']
survey_questions = ['Dependable', 'Reliable', 'Unresponsive', 'Predictable', 'Act consistently', 'Function successfully', 'Malfunction', 'Have errors', 'Provide feedback', 'Meet the needs of the mission/task', 'Provide appropriate information', 'Communicate with people', 'Perform exactly as instructed', 'Follow directions']

survey_leader = [a+s for s in survey_questions for a in ["leader_", "nonleader_"]]

demographic_questions = ['Age', 'How often do you play first-person video games on average?', 'How many times have you interacted with the TeamCollab simulation environment in the past?', 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?']
ptt_questions = ['Generally, I trust automated agents', 'Automated agents help me solve many problems', 'I think it’s a good idea to rely on automated agents for help', 'I don’t trust the information I get from automated agents', 'Automated agents are reliable', 'I rely on automated agents']

translation = {'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?': "LLM_Exp", 'How many times have you interacted with the TeamCollab simulation environment in the past?': "Simulation_Exp", 'How often do you play first-person video games on average?': "VideoGame_Exp", 'Performance: How successful were you in accomplishing what you were asked to do?': "Performance_P", 'Generally, I trust automated agents': "Trust_Agents", 'Provide appropriate information': "Appropriate_Info", "team_carrying_composition_AI": "Carry_Composition", "danger_weight_coverage": "Weight_Coverage", "communication_range_error": "Range_Error", 'Perform exactly as instructed': "Perform_Exactly", "misunderstanding_rate": "Misunderstand", "dispersion_to_leader": "Dispersion_Leader", "completed_orders": "Completed_Orders", "sensed_0": "Sensed_0", "sensed_4": "Sensed_4", "sensed_2": "Sensed_2", "redundancy":"Redundancy", "underspecified_rate":"Underspecified", "completed_other_type": "Completed_Others", "precision": "Precision", "repeated_orders":"Repeated_Orders", "change_rate": "Change_Rate", "Follow directions": "Follow_Directions", "Act consistently": "Act_Consistently", "team_dispersion": "Team_Dispersion", "total_orders": "Total_Orders", "path_eff_median": "Carry_Efficiency", "latency_norm_median": "Effort_Latency", "completion_rate": "Completion_Rate", "Unresponsive": "Responsive"}


table_explanation = {
"LLM_Exp": 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?',
"Simulation_Exp":'How many times have you interacted with the TeamCollab simulation environment in the past?',
"VideoGame_Exp": 'How often do you play first-person video games on average?',
"Performance_P": 'Performance: How successful were you in accomplishing what you were asked to do?',
"Trust_Agents": 'Generally, I trust automated agents',
"Appropriate_Info": 'Provide appropriate information',
"Perform_Exactly": 'Perform exactly as instructed',
"Follow_Directions": "Follow directions",
"Act_Consistently": "Act consistently",
"Carry_Composition": "Average fraction of AIs to humans across carrying groups",
"Weight_Coverage": "Proportion of total dangerous weight that was successfully carried",
"Range_Error": "Number of times an order failed to reach its intended recipient",
"Misunderstand": "Rate of AI misinterpretations, calculated as misunderstood orders divided by total orders",
"Dispersion_Leader": "Mean distance between the leader and their designated partners, averaged across the session",
"Completed_Orders": "Total count of successfully completed orders",
"Sensed_0": "Proportion of objects that were never sensed", 
"Sensed_4": "Proportion of objects sensed by all four agents", 
"Sensed_2": "Proportion of objects sensed by exactly two agents",
"Redundancy": "Average number of extra sensings per object beyond the first",
"Underspecified": "Proportion of orders issued with incomplete team composition or missing objective information",
"Completed_Others": "Proportion of orders completed by agents of the non-leader type",
"Precision": "Proportion of carried objects that were dangerous", 
"Repeated_Orders": "Number of times an issued order was repeated",
"Change_Rate": "Number of times orders were modified or reassigned for each agent",
"Team_Dispersion": "Mean pairwise distance between all agents, averaged across the session",
"Total_Orders": "Total count of issued orders",
"Carry_Efficiency": "Median time-per-distance efficiency for carried dangerous objects", 
"Effort_Latency": "Median latency from first order to first carry, normalized by distance × weight", 
"Completion_Rate": "Proportion of successfully completed orders out of all issued orders"
}


def spearman_permutation_test(x, y, n_resamples=10000, alternative="two-sided", seed=None):
    """
    Vectorized permutation test for Spearman's rho.
    Works fast for small n (e.g., n=16) and large resample counts.

    Parameters
    ----------
    x, y : array-like
        Paired samples.
    n_resamples : int
        Number of permutations.
    alternative : {"two-sided", "greater", "less"}
        Test alternative.
    seed : int or None
        RNG seed.

    Returns
    -------
    rho_obs : float
        Observed Spearman correlation.
    p_value : float
        Permutation p-value.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Drop NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = x.size
    if n < 3:
        return np.nan, np.nan  # too few points for a meaningful test

    # Rank once (ties handled with average ranks)
    xr = rankdata(x, method="average")
    yr = rankdata(y, method="average")

    # Z-score ranks (population std is fine; cancels in correlation)
    xz = (xr - xr.mean()) / xr.std(ddof=0)
    yz = (yr - yr.mean()) / yr.std(ddof=0)

    # Observed Spearman (Pearson on z-scored ranks)
    rho_obs = float(np.mean(xz * yz))

    # Generate all permutations at once using argsort of random noise
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_resamples, n)), axis=1)   # shape (B, n)

    # Apply permutations to yz only (Spearman ranks of y just permute)
    yz_perm = yz[perms]                                       # shape (B, n)

    # Correlations for all permutations (vectorized)
    rhos = np.mean(yz_perm * xz, axis=1)                      # shape (B,)

    # Compute p-value based on alternative
    if alternative == "two-sided":
        p = float(np.mean(np.abs(rhos) >= abs(rho_obs)))
    elif alternative == "greater":
        p = float(np.mean(rhos >= rho_obs))
    elif alternative == "less":
        p = float(np.mean(rhos <= rho_obs))
    else:
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")

    return rho_obs, p
    
def analyze_survey_significance(
    df: pd.DataFrame,
    survey_keys: Iterable[str],
    id_col: str = "team_id",
    session_col: str = "session",
    leader_col: str = "leader_type",
    alpha: float = 0.05,
    save_csv: bool = False,
    outdir: str = ".",
) -> Dict[str, pd.DataFrame]:
    """
    Run significance tests on survey items:
      1) Within-team differences across sessions (Friedman).
      2) Between leader types at each session (Kruskal).
      3) If Kruskal significant (p_adj < alpha), run pairwise Mann–Whitney post-hoc with Holm correction.

    Returns a dict with DataFrames:
      - 'friedman': per-question omnibus across sessions
      - 'kruskal': per-question, per-session omnibus across leader types
      - 'pairwise': per-question, per-session pairwise leader-type comparisons (only rows tested)

    Parameters
    ----------
    df : DataFrame with columns [id_col, session_col, leader_col] + survey_keys
    survey_keys : iterable of column names for survey questions
    id_col, session_col, leader_col : column names
    alpha : significance level for post-hoc triggering (after Holm correction)
    save_csv : if True, writes CSVs to `outdir`
    outdir : directory for CSVs
    """

    # -----------------------------
    # Helpers
    # -----------------------------
    def _kendalls_w_from_friedman(chi2: float, n: int, k: int) -> float:
        # W = chi2 / (n * (k - 1))
        if n is None or k is None or n <= 0 or k is None or k <= 1 or not np.isfinite(chi2):
            return np.nan
        return chi2 / (n * (k - 1))

    def _epsilon_sq_from_kruskal(H: float, n_total: int, k_groups: int) -> float:
        # ε² = (H - k + 1) / (n - k)
        if any(v is None for v in [H, n_total, k_groups]) or (n_total - k_groups) <= 0 or not np.isfinite(H):
            return np.nan
        return max(0.0, (H - k_groups + 1) / (n_total - k_groups))

    def _rank_biserial_from_mwu(U: float, n1: int, n2: int) -> float:
        # r_rb = 1 - 2U/(n1*n2) ; ranges [-1,1]
        if n1 <= 0 or n2 <= 0 or not np.isfinite(U):
            return np.nan
        return 1.0 - (2.0 * U) / (n1 * n2)

    # -----------------------------
    # 1) Friedman (within-team across sessions)
    # -----------------------------
    friedman_rows = []
    for q in survey_keys:
        # wide: index=id, columns=session
        wide = df.pivot_table(index=id_col, columns=session_col, values=q, aggfunc="mean")
        wide = wide.dropna(axis=0, how="any")  # need complete cases across sessions
        k = wide.shape[1]
        n = wide.shape[0]
        if n >= 2 and k >= 2:
            try:
                stat, p = friedmanchisquare(*[wide[c] for c in wide.columns])
            except Exception:
                stat, p = np.nan, np.nan
            W = _kendalls_w_from_friedman(stat, n, k)
        else:
            stat, p, W = np.nan, np.nan, np.nan

        friedman_rows.append({
            "question": q,
            "n_subjects": n,
            "k_sessions": k,
            "statistic": stat,
            "p_value": p,
            "kendalls_W": W,
        })

    friedman_df = pd.DataFrame(friedman_rows)
    # Holm correction across questions (one family)
    friedman_df["p_adj"] = multipletests(friedman_df["p_value"], method="holm")[1]

    # -----------------------------
    # 2) Kruskal (between leader types at each session)
    # -----------------------------
    kruskal_rows = []
    sessions_sorted = sorted(df[session_col].dropna().unique())
    for q in survey_keys:
        for s in sessions_sorted:
            d = df[df[session_col] == s]
            # keep only leader groups with at least 2 valid observations
            groups = []
            group_names = []
            for lt, d_g in d.groupby(leader_col):
                vals = d_g[q].dropna()
                if len(vals) >= 2:
                    groups.append(vals)
                    group_names.append(lt)

            if len(groups) >= 2:
                try:
                    stat, p = kruskal(*groups)
                except Exception:
                    stat, p = np.nan, np.nan
                n_total = int(sum(len(g) for g in groups))
                k_groups = len(groups)
                eps2 = _epsilon_sq_from_kruskal(stat, n_total, k_groups)
            else:
                stat, p, eps2, n_total, k_groups = np.nan, np.nan, np.nan, sum(len(g) for g in groups) if groups else 0, len(groups)

            kruskal_rows.append({
                "question": q,
                session_col: s,
                "k_groups": len(groups),
                "n_total": int(sum(len(g) for g in groups)) if groups else 0,
                "statistic": stat,
                "p_value": p,
                "epsilon_sq": eps2,
            })

    kruskal_df = pd.DataFrame(kruskal_rows)
    # Holm correction per session (one family per session across questions)
    kruskal_df["p_adj"] = np.nan
    for s in sessions_sorted:
        mask = kruskal_df[session_col] == s
        if mask.sum() > 0:
            pvals = kruskal_df.loc[mask, "p_value"].astype(float)
            kruskal_df.loc[mask, "p_adj"] = multipletests(pvals, method="holm")[1]

    # -----------------------------
    # 3) Post-hoc pairwise (Mann–Whitney) when Kruskal is significant after correction
    # -----------------------------
    pairwise_rows = []
    # Pre-scan: for each (q,s) where kruskal p_adj < alpha, run pairwise among available groups
    sig_mask = (kruskal_df["p_adj"].astype(float) < alpha)
    to_test = kruskal_df.loc[sig_mask, ["question", session_col]]

    for _, row in to_test.iterrows():
        q = row["question"]
        s = row[session_col]
        d = df[df[session_col] == s][[leader_col, q]].dropna()
        # Groups with at least 2 observations
        groups = {lt: grp[q].dropna().values for lt, grp in d.groupby(leader_col)}
        groups = {k: v for k, v in groups.items() if len(v) >= 2}
        if len(groups) < 2:
            continue

        # All pairwise comparisons
        pairs = list(combinations(sorted(groups.keys()), 2))
        pvals = []
        stats_tmp: Dict[Tuple[str, str], Tuple[float, float, float]] = {}

        for a, b in pairs:
            x, y = groups[a], groups[b]
            # two-sided MWU with 'auto' method (scipy >=1.7)
            try:
                res = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                U = res.statistic
                p = res.pvalue
            except TypeError:
                # fallback for older scipy
                U, p = mannwhitneyu(x, y, alternative="two-sided")
            r_rb = _rank_biserial_from_mwu(U, len(x), len(y))
            pvals.append(p)
            stats_tmp[(a, b)] = (U, p, r_rb)

        # Holm correction within this (q,s) family
        _, pvals_adj, _, _ = multipletests(pvals, method="holm")
        for (a, b), p_adj in zip(pairs, pvals_adj):
            U, p_raw, r_rb = stats_tmp[(a, b)]
            pairwise_rows.append({
                "question": q,
                session_col: s,
                "group_a": a,
                "group_b": b,
                "n_a": len(groups[a]),
                "n_b": len(groups[b]),
                "U": U,
                "p_value": p_raw,
                "p_adj": p_adj,
                "rank_biserial": r_rb,
            })

    pairwise_df = pd.DataFrame(pairwise_rows) if pairwise_rows else pd.DataFrame(
        columns=["question", session_col, "group_a", "group_b", "n_a", "n_b", "U", "p_value", "p_adj", "rank_biserial"]
    )

    # -----------------------------
    # Optional CSV export
    # -----------------------------
    if save_csv:
        friedman_df.to_csv(f"{outdir}/friedman_session_differences.csv", index=False)
        kruskal_df.to_csv(f"{outdir}/kruskal_leader_differences.csv", index=False)
        pairwise_df.to_csv(f"{outdir}/posthoc_pairwise_leader_differences.csv", index=False)

    return {
        "friedman": friedman_df.sort_values("p_adj"),
        "kruskal": kruskal_df.sort_values([session_col, "p_adj"]),
        "pairwise": pairwise_df.sort_values([session_col, "question", "p_adj"]) if not pairwise_df.empty else pairwise_df
    }



#
# Correlations with transitions between AI to human, human to AI, AI to AI, etc.
#

def _normalize_leader(val):
    if pd.isna(val): return np.nan
    if isinstance(val, (int, np.integer)):
        return "AI" if int(val) == 1 else "Human"
    s = str(val).strip().lower()
    if s in ("ai","a","robot","bot","1","ai_leader","ai-led","ai led"): return "AI"
    if s in ("human","h","0","human_leader","human-led","human led"):    return "Human"
    if s in ("true","false"): return "AI" if s == "true" else "Human"
    if "ai" in s: return "AI"
    if "human" in s: return "Human"
    raise ValueError(f"Unrecognized leader value: {val!r}")

def add_transition_columns(df, team_col="team_id", session_col="session",
                           leader_col="leader_type", perf_col="performance",
                           feature_cols=None, compute_feature_deltas=False):
    """
    Adds: leader_norm, prev_leader, prev_performance, is_contiguous, transition, delta_performance.
    If compute_feature_deltas=True, also adds delta_<feature> for each feature in feature_cols.
    """
    df = df.copy()
    df = df.sort_values([team_col, session_col])
    df["leader_norm"] = df[leader_col].map(_normalize_leader)
    df["prev_leader"] = df.groupby(team_col)["leader_norm"].shift(1)
    df["prev_performance"] = df.groupby(team_col)[perf_col].shift(1)

    # contiguity: next session exactly +1; adjust if you use dates or different rule
    df["is_contiguous"] = df.groupby(team_col)[session_col].diff().eq(1)

    df["transition"] = np.where(
        df["is_contiguous"],
        df["prev_leader"].fillna("NA") + "->" + df["leader_norm"].fillna("NA"),
        np.nan
    )

    df["delta_performance"] = df[perf_col] - df["prev_performance"]

    if compute_feature_deltas and feature_cols:
        for f in feature_cols:
            # compute shift per team, then wipe non-contiguous deltas to NaN
            d = df[f] - df.groupby(team_col)[f].shift(1)
            df[f"delta_{f}"] = d.where(df["is_contiguous"], np.nan)

    return df

def compute_correlations(df, y_col, feature_cols):
    rows = []
    for feat in feature_cols:
        # only numeric features
        if not pd.api.types.is_numeric_dtype(df[feat]):
            continue
        sub = df[[feat, y_col]].dropna()
        n = len(sub)
        if n < 3:
            rows.append(dict(feature=feat, N=n, pearson_r=np.nan, pearson_p=np.nan,
                             spearman_rho=np.nan, spearman_p=np.nan))
            continue
        r, rp = stats.pearsonr(sub[feat], sub[y_col])
        rho, rhop = stats.spearmanr(sub[feat], sub[y_col])
        rows.append(dict(feature=feat, N=n, pearson_r=r, pearson_p=rp,
                         spearman_rho=rho, spearman_p=rhop))
    out = pd.DataFrame(rows).sort_values(["pearson_p","spearman_p","feature"])
    return out

def correlations_by_transition(df,
                               team_col="team_id", session_col="session",
                               leader_col="leader_type", perf_col="performance",
                               use_delta=False,  # when True: Δfeature vs Δperformance
                               drop_non_numeric=True):
    """
    If use_delta=False: correlates raw features with performance.
    If use_delta=True:  correlates Δfeatures with Δperformance (contiguous only).
    Uses your build_features(df) output directly (no merge).
    """
    # 1) Build features with your pipeline (keeps original columns)
    feat_df = df.copy()

    # 2) Determine the feature set produced by build_features
    exclude = {team_col, session_col, leader_col, perf_col,
               "leader_norm", "prev_leader", "prev_performance",
               "is_contiguous", "transition", "delta_performance"}
    feature_cols = [c for c in feat_df.columns if c not in exclude]

    # (optional) keep only numeric features up-front
    if drop_non_numeric:
        feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(feat_df[c])]

    # 3) Add transitions (+ feature deltas only if we will use them)
    df_t = add_transition_columns(
        feat_df,
        team_col=team_col, session_col=session_col,
        leader_col=leader_col, perf_col=perf_col,
        feature_cols=feature_cols,
        compute_feature_deltas=use_delta
    )

    # 4) Choose outcome & feature list for correlation
    if use_delta:
        y_col = "delta_performance"
        corr_features = [f"delta_{f}" for f in feature_cols if f"delta_{f}" in df_t.columns]
    else:
        y_col = perf_col
        corr_features = feature_cols

    # 5) Compute per-transition correlations
    results = {}
    for trans in ["AI->Human", "Human->AI", "AI->AI", "Human->Human"]:
        mask = df_t["transition"].eq(trans)
        results[trans] = compute_correlations(df_t.loc[mask], y_col, corr_features)

    return results



def save_latex_tables(results, filename_prefix="corr_tables", top_n=10, caption_suffix="performance"):
    """
    Save LaTeX tables for each transition type with the top N correlated features.
    results: dict from correlations_by_transition
    """
    for trans, df in results.items():
        if df.empty:
            continue
        # Take top N by absolute Pearson r
        df_top = df.reindex(df['pearson_r'].abs().sort_values(ascending=False).index).head(top_n)

        # LaTeX table
        latex_str = r"""\begin{table}[htbp]
\centering
\caption{Top correlations with %s for transition %s}
\label{tab:corr_%s}
\begin{tabular}{lrrrrr}
\toprule
Feature & $N$ & $r_{pearson}$ & $p_{pearson}$ & $\rho_{spearman}$ & $p_{spearman}$ \\
\midrule
""" % (caption_suffix, trans, trans.replace("->","to"))

        for _, row in df_top.iterrows():
            latex_str += f"{row['feature']} & {int(row['N'])} & {row['pearson_r']:.3f} & {row['pearson_p']:.3g} & {row['spearman_rho']:.3f} & {row['spearman_p']:.3g} \\\\\n"

        latex_str += r"""\bottomrule
\end{tabular}
\end{table}
"""

        with open(f"{filename_prefix}_{trans.replace('->','to')}.tex", "w") as f:
            f.write(latex_str)


def _detect_corr_col_trans(df: pd.DataFrame) -> str:
    """Pick the correlation column; prefer common names."""
    preferred = ["corr_with_Δperf", "corr_with_delta_perf", "corr_with_performance", "corr", "r", "rho"]
    for c in preferred:
        if c in df.columns:
            return c
    # fallback: first numeric column that isn't 'abs_corr'
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found in provided DataFrame.")

def _detect_corr_col_trans2(df: pd.DataFrame) -> str:
    """Always use Spearman's rho for transition correlation tables."""
    if "spearman_rho" not in df.columns:
        raise ValueError(
            f"'spearman_rho' not found in DataFrame. Available columns: {list(df.columns)}"
        )
    return "spearman_rho"


def _labelify(s: str) -> str:
    """Make a safe LaTeX label suffix from a transition string."""
    return s.replace(" ", "_").replace(">", "to").replace("/", "-")

def write_by_transition_latex_with_meta(
    corr_by_transition: Dict[str, pd.DataFrame],
    out_tex: str = "transition_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 50,
    abs_thresh: Optional[float] = None,     # filter inside single-transition & ALL-groups meta
    # Pairwise knobs:
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.0,          # strong in BOTH transitions means |r|>=this in each
    diff_thresh_pair: float = 0.40,         # if same sign but |Δr|>=this → dissimilar
    top_k_pair: int = 50,
    pretty_names: Optional[Dict[str, str]] = None,
    strength_thresh: int = 0.4  # map transition key to pretty caption text
):
    """
    corr_by_transition: dict like {
        "AI->Human": DataFrame(feature, corr_with_*),
        "Human->AI": ...,
        "AI->AI": ...,
        "Human->Human": ...
    }

    Produces:
      • One LaTeX table per transition
      • ALL-transitions meta tables: Top shared (min |r|), Top dissimilar (range of r)
      • Pairwise tables for every transition pair: Shared & Dissimilar (strong in both)
    """
    # ---------------- 1) Per-transition tables ----------------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with performance by transition\n\n")
        for trans, tbl in corr_by_transition.items():
            name = pretty_names.get(trans, trans) if pretty_names else trans
            df = tbl.copy()
            corr_col = _detect_corr_col_trans2(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Transition {name} --------\n")
            f.write(df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Correlations with performance — Transition {name} ({note_excluded})",
                label=f"tab:trans_corr_{_labelify(name)}"
            ))
            f.write("\n\n")

    # ---------------- 2) Long-form for ALL-transitions meta & pairwise ----------------
    long_rows = []
    for trans, tbl in corr_by_transition.items():
        name = pretty_names.get(trans, trans) if pretty_names else trans
        df = tbl.copy()
        corr_col = _detect_corr_col_trans2(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "transition": name,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
            })
    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    transitions = sorted(longdf["transition"].unique().tolist())

    # ---------------- 3) ALL-transitions meta ----------------
    meta_df = longdf.copy()
    if abs_thresh is not None:
        meta_df = meta_df[meta_df["abs_corr"] >= abs_thresh]

    agg = (
        meta_df.groupby("feature")
               .agg(n_trans=("transition","nunique"),
                    mean_corr=("corr","mean"),
                    mean_abs=("abs_corr","mean"),
                    min_abs=("abs_corr","min"),
                    max_abs=("abs_corr","max"),
                    min_corr=("corr","min"),
                    max_corr=("corr","max"))
               .reset_index()
    )
    shared_all = agg.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    agg["range_corr"] = agg["max_corr"] - agg["min_corr"]
    dissim_all = agg.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_trans","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_trans","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- ALL TRANSITIONS: Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across transitions "
                     f"(ranked by minimum |r| across transitions; {note_excluded})."),
            label="tab:trans_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- ALL TRANSITIONS: Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across transitions "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:trans_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------------- 4) Pairwise transition comparisons ----------------
    if not do_pairwise or len(transitions) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN TRANSITIONS --------\n\n")

    # Pre-filtered per-transition for pairwise logic (strong features only)
    per_trans_df = {}
    for T in transitions:
        sub = longdf[longdf["transition"] == T].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_trans_df[T] = sub.set_index("feature")[["corr","abs_corr"]]

    for A, B in itertools.combinations(transitions, 2):
        A_df = per_trans_df[A]
        B_df = per_trans_df[B]
        if A_df.empty or B_df.empty:
            continue

        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
              
        if inter.empty:
            continue

        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        # Shared (same sign, strong in both)
        shared = inter[inter["same_sign"]].sort_values(
            ["delta_r", "min_abs"], ascending=[True, False]
        ).head(top_k_pair).copy()

        # Dissimilar (opposite sign OR large Δ in magnitude)
        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(
            ["delta_r","min_abs"], ascending=[False, False]
        ).head(top_k_pair).copy()

        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})",
                                   "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Transitions {A} vs {B}: shared strong correlations "
                         f"(|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:trans_pair_shared_{_labelify(A)}_vs_{_labelify(B)}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Transitions {A} vs {B}: dissimilar strong correlations "
                         f"(|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:trans_pair_dissimilar_{_labelify(A)}_vs_{_labelify(B)}"
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise transition comparisons) to: {out_tex}")
    

# ---------- helpers ----------
def fisher_pdiff_session(r1: float, n1: int, r2: float, n2: int) -> Tuple[float, float]:
    """
    Fisher r-to-z comparison of two independent correlations.
    Returns (z_diff, p_two_sided).
    """
    # guard boundaries
    r1 = np.clip(r1, -0.999999, 0.999999)
    r2 = np.clip(r2, -0.999999, 0.999999)
    if (n1 is None) or (n2 is None) or (n1 < 4) or (n2 < 4):
        return (np.nan, np.nan)
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1/(n1 - 3) + 1/(n2 - 3))
    if not np.isfinite(se) or se == 0:
        return (np.nan, np.nan)
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return float(z), float(p)

def _labelify_session(s: int) -> str:
    return f"s{s}"

def _pick_feature_cols_for_session(df: pd.DataFrame,
                                   exclude_like=("team_id","session","leader_type","layout","suggestions","name","performance")) -> List[str]:
    """
    Pick numeric feature columns for within-session correlation (raw features).
    Excludes id/meta columns and the target 'performance' itself.
    """
    bad = set(exclude_like)
    numerics = []
    for c in df.columns:
        if c in bad: 
            continue
        if c.startswith("Δ_"):  # exclude any delta leftovers
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numerics.append(c)
    return numerics

# ---------- main: session-based ----------
def write_session_corr_latex_with_meta(
    res_any: dict,
    out_tex: str = "session_corr_tables.tex",
    note_excluded: str = "per-session correlations; excludes ids & Δ_*",
    # feature exclusion (optional, e.g., to drop tautological measures)
    exclude_features: List[str] = None,
    # single-session display filter
    abs_thresh_single: float = None,      # e.g., 0.30
    top_k_meta: int = 10,
    # pairwise gates
    abs_thresh_pair: float = 0.30,        # keep features with |r|>= in BOTH sessions before testing
    strength_thresh: float = 0.40,        # additional strength gate for pair tables
    diff_thresh_pair: float = 0.40,       # count as dissimilar if same sign but |Δr|>= this
    top_k_pair: int = 10
):
    """
    Builds LaTeX tables for per-session correlations of raw 'performance' vs raw features,
    plus meta (shared/dissimilar across sessions) and pairwise session comparisons with
    Fisher z-tests (p_diff) and FDR (q_diff).

    Input:
      res_any: dictionary that MUST contain 'feat_df' (from your build_features(...) output).
               It can be the same dict you used before (raw_delta_descriptives_flexible(...))
               since that also returns 'feat_df'.

    Output:
      Writes LaTeX to `out_tex` with:
        1) Per-session correlation tables: feature, r, p, n
        2) ALL-sessions meta: Top shared (by min|r|) and Top dissimilar (by range r)
        3) Pairwise session comparisons: Shared & Dissimilar (with delta_r, z_diff, p_diff, q_diff)
    """
    # ---------------------------
    # 0) get the features dataframe
    # ---------------------------
    if "feat_df" not in res_any:
        raise ValueError("res_any must contain 'feat_df' with raw sessions (team_id, session, leader_type, performance, features...).")
    feat_df = res_any["feat_df"].copy()

    # basic guards
    if feat_df.empty or "performance" not in feat_df.columns or "session" not in feat_df.columns:
        with open(out_tex, "w") as f:
            f.write("% No usable rows/columns in feat_df for session-level correlations.\n")
        print(f"Wrote LaTeX (empty): {out_tex}")
        return

    if exclude_features is None:
        exclude_features = []

    # ---------------------------
    # 1) per-session correlations (Spearman)
    # ---------------------------
    corr_by_session: Dict[int, pd.DataFrame] = {}
    sessions = sorted(pd.unique(feat_df["session"]).tolist())
    for s in sessions:
        sub = feat_df[feat_df["session"] == s].copy()
        if sub.empty:
            corr_by_session[s] = pd.DataFrame(columns=["feature","r","p","n"])
            continue

        fcols = _pick_feature_cols_for_session(sub)
        fcols = [c for c in fcols if c not in set(exclude_features)]

        rows: List[dict] = []
        for feat in fcols:
            xy = sub[[feat, "performance"]].dropna()
            if xy.shape[0] < 3 or xy[feat].nunique() <= 1 or xy["performance"].nunique() <= 1:
                continue
            r, p = spearmanr(xy[feat], xy["performance"])
            r, p = spearman_permutation_test(xy[feat], xy["performance"], n_resamples=50000, seed=42)
            rows.append({"feature": feat, "r": float(r), "p": float(p), "n": int(len(xy))})

        df_stats = pd.DataFrame(rows)
        if not df_stats.empty:
            if abs_thresh_single is not None:
                df_stats = df_stats.assign(abs_r=df_stats["r"].abs())
                df_stats = df_stats[df_stats["abs_r"] >= abs_thresh_single].drop(columns="abs_r")
            df_stats = df_stats.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        corr_by_session[s] = df_stats if not df_stats.empty else pd.DataFrame(columns=["feature","r","p","n"])

    # write per-session LaTeX
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations of performance with features by session\n\n")
        for s in sessions:
            df = corr_by_session.get(s, pd.DataFrame()).copy()
            if df.empty:
                continue
            if abs_thresh_single is not None:
                df = df.assign(abs_r=df["r"].abs())
                df = df[df["abs_r"] >= abs_thresh_single].drop(columns="abs_r")
            df = df.sort_values("r", key=lambda s: s.abs(), ascending=False)
            f.write(f"%% -------- Session {s} --------\n")
            f.write(df[["feature","r","p","n"]].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=f"Session {s}: Spearman correlations with performance ({note_excluded}).",
                label=f"tab:session_corr_{_labelify_session(int(s))}"
            ))
            f.write("\n\n")

    # ---------------------------
    # 2) ALL-sessions meta (shared/dissimilar across sessions)
    # ---------------------------
    long_rows = []
    for s, tbl in corr_by_session.items():
        if tbl is None or tbl.empty:
            continue
        d = tbl.copy()
        d = d[pd.to_numeric(d["r"], errors="coerce").notna()]
        for r in d.itertuples(index=False):
            long_rows.append({"session": s, "feature": getattr(r, "feature"), "corr": float(getattr(r, "r")), "p": float(getattr(r,"p"))})

    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    meta_df = longdf if abs_thresh_single is None else longdf[longdf["abs_corr"] >= abs_thresh_single]

    agg = (meta_df.groupby("feature")
                 .agg(n_sess=("session","nunique"),
                      mean_corr=("corr","mean"),
                      mean_abs=("abs_corr","mean"),
                      min_abs=("abs_corr","min"),
                      max_abs=("abs_corr","max"),
                      min_corr=("corr","min"),
                      max_corr=("corr","max"))
                 .reset_index())

    shared_all = agg.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k_meta).copy()
    agg["range_corr"] = agg["max_corr"] - agg["min_corr"]
    dissim_all = agg.sort_values("range_corr", ascending=False).head(top_k_meta).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    with open(out_tex, "a") as f:
        f.write("%% -------- ALL SESSIONS: Top shared features --------\n")
        f.write(shared_all[["feature","n_sess","min_abs","mean_abs","max_abs","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} features shared across sessions "
                     f"(ranked by minimum |r| across sessions; {note_excluded})."),
            label="tab:session_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- ALL SESSIONS: Top dissimilar features --------\n")
        f.write(dissim_all[["feature","n_sess","range_corr","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} most dissimilar features across sessions "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:session_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------------------------
    # 2b) OVERALL ("ALL") correlations across all sessions
    #     Using the SAME variables already present in longdf
    # ---------------------------
    features_seen = sorted(longdf["feature"].unique().tolist())
    overall_long = _overall_corr_long(
        feat_df=feat_df,
        features=features_seen,
        exclude_features=exclude_features
    )

    # Combine per-session + ALL
    longdf_with_all = pd.concat([longdf, overall_long], ignore_index=True)

    '''
    # (Optional) write a small ALL-sessions table to LaTeX as well
    if not overall_long.empty:
        with open(out_tex, "a") as f:
            f.write("%% -------- OVERALL (ALL sessions pooled): correlations --------\n")
            f.write(overall_long.sort_values("corr", key=lambda s: s.abs(), ascending=False)
                    .assign(corr=lambda d: d["corr"].round(3))[["feature","corr"]]
                    .to_latex(index=False, float_format="%.3f",
                              caption=(f"Spearman correlations with performance pooling all sessions "
                                       f"({note_excluded})."),
                              label="tab:session_overall_corr"))
            f.write("\n\n")
    '''
    # ---------------------------
    # 3 Pairwise session comparisons (Fisher z test + FDR)
    # ---------------------------
    
    features_to_plot = {}
    non_small_p_diff = []
    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN SESSIONS (with p_diff & q_diff) --------\n\n")

    # Build dict per session keyed by feature
    per_sess = {}
    for s in sessions:
        dfS = corr_by_session.get(s, pd.DataFrame())
        if dfS is None or dfS.empty:
            continue
        dfS = dfS.copy()
        dfS["abs_r"] = dfS["r"].abs()
        if abs_thresh_pair is not None:
            dfS = dfS[dfS["abs_r"] >= abs_thresh_pair]
        per_sess[s] = dfS.set_index("feature")[["r","p","n","abs_r"]]

    for A, B in itertools.combinations(sessions, 2):
        if A not in per_sess or B not in per_sess:
            continue
        A_df, B_df = per_sess[A], per_sess[B]
        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        if inter.empty:
            continue

        # require strong enough in BOTH sessions, then strength gate
        inter = inter[(inter["abs_r_A"] >= strength_thresh) & (inter["abs_r_B"] >= strength_thresh)]
        if inter.empty:
            continue

        # Fisher tests per feature
        zvals, pdiffs = [], []
        for feat, r in inter.iterrows():
            z_diff, p_diff = fisher_pdiff_session(float(r["r_A"]), int(r["n_A"]),
                                          float(r["r_B"]), int(r["n_B"]))
            zvals.append(z_diff); pdiffs.append(p_diff)

        out = (inter.assign(
                    delta_r = inter["r_B"] - inter["r_A"],
                    abs_delta_r = (inter["r_B"] - inter["r_A"]).abs(),
                    same_sign = np.sign(inter["r_A"]) == np.sign(inter["r_B"]),
                    min_abs = np.minimum(inter["abs_r_A"], inter["abs_r_B"]),
                    z_diff = zvals,
                    p_diff = pdiffs)
                    .reset_index().rename(columns={"index":"feature"}))

        # FDR within this A vs B comparison
        mask = out["p_diff"].notna()
        if mask.any():
            _, q, _, _ = multipletests(out.loc[mask, "p_diff"], method="fdr_bh")
            out.loc[mask, "q_diff"] = q
        else:
            out["q_diff"] = np.nan

        # shared & dissimilar panes
        shared = out[out["same_sign"]].sort_values(["abs_delta_r","min_abs"], ascending=[True, False]).head(top_k_pair)
        dissim_mask = (out["abs_delta_r"] >= diff_thresh_pair) & ((out["abs_r_A"] >= diff_thresh_pair) | (out["abs_r_B"] >= diff_thresh_pair))
        dissim = out[dissim_mask].sort_values(["abs_delta_r","min_abs"], ascending=[False, False]).head(top_k_pair)
        ignore = ["Age",'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?', 'Performance: How successful were you in accomplishing what you were asked to do?', "benign_ratio", "frac_dangerous_ambiguous", "repeated_orders", "reliability_score"]
        ignore = ["Age",'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?', 'Performance: How successful were you in accomplishing what you were asked to do?', "benign_ratio", "frac_dangerous_ambiguous"]
        
        dissim_mask2 = dissim["p_diff"] <= 0.05
        dissim_final_list = dissim[dissim_mask2]['feature'].values.tolist()
        for ff in dissim["feature"].values.tolist():
        
            if ff in dissim_final_list and ff not in non_small_p_diff:
                non_small_p_diff.append(ff)        

            if ff not in ignore:
                if ff not in features_to_plot.keys():
                    features_to_plot[ff] = 0
                features_to_plot[ff] += 1


        cols = ["feature","r_A","p_A","n_A","r_B","p_B","n_B","delta_r","z_diff","p_diff","q_diff"]
        with open(out_tex, "a") as f:
            f.write(f"%% -------- SESSIONS: {A} vs {B} — Shared (same sign) --------\n")
            f.write(shared[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Sessions {A} vs {B}: shared-direction correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:sess_pair_shared_{_labelify_session(int(A))}_vs_{_labelify_session(int(B))}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- SESSIONS: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Sessions {A} vs {B}: dissimilar correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:sess_pair_dissimilar_{_labelify_session(int(A))}_vs_{_labelify_session(int(B))}"
            ))
            f.write("\n\n")

    
    features_to_plot_final = list(features_to_plot.keys())
    #features_to_plot_final = [d for d in non_small_p_diff]
    

    print(f"Wrote LaTeX tables (per-session, meta, pairwise with Fisher tests) to: {out_tex}")
    #plot_correlations_over_sessions(longdf, features=features_to_plot_final,out_png="figure_corr_trajectories.png", by="session")
    #plot_sankey_style_correlations(longdf, features=features_to_plot_final)
    

    survey_questions = ['Dependable', 'Reliable', 'Unresponsive', 'Predictable', 'Act consistently', 'Function successfully', 'Malfunction', 'Have errors', 'Provide feedback', 'Meet the needs of the mission/task', 'Provide appropriate information', 'Communicate with people', 'Perform exactly as instructed', 'Follow directions']
    demographic_questions = ['Age', 'How often do you play first-person video games on average?', 'How many times have you interacted with the TeamCollab simulation environment in the past?', 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?']
    
    prefixes = ["leader_","nonleader_"]
    features_to_remove = [ *[prefix + q for prefix in prefixes for q in ["Age"]], "Age", 'leader_I think it’s a good idea to rely on automated agents for help', 'nonleader_Provide appropriate information', "total_orders", 'Unresponsive', "nonleader_Meet the needs of the mission/task", "reliability_score"]

    for ftr in features_to_remove:
        if ftr in features_to_plot_final:
            features_to_plot_final.remove(ftr)

    features_to_plot_final.extend(["underspecified_rate","danger_weight_coverage"])
    
    '''
    ###Remove this
    longdf_with_all = longdf_with_all[longdf_with_all["feature"].isin(features_to_plot_final)].copy()
    # Function to translate with prefix handling
    def translate_with_prefix(name, translation):
        prefix = ""
        if name.startswith("leader_"):
            prefix = "leader_"
            base = name[len("leader_"):]
        elif name.startswith("nonleader_"):
            prefix = "nonleader_"
            base = name[len("nonleader_"):]
        else:
            base = name

        # Translate base if in dictionary
        translated_base = translation.get(base, base)
        return prefix + translated_base

    longdf_with_all["feature"] = longdf_with_all["feature"].apply(lambda x: translate_with_prefix(x, translation))

    # Save to CSV
    longdf_with_all.to_csv("translated_features_session.csv", index=False)
    '''
    longdf_with_all["session"] = longdf_with_all["session"].replace({i:i+1 for i in range(4)})
    
    pdb.set_trace()
    #pdb.set_trace()
    #longdf_with_all[longdf_with_all["feature"].isin(features_to_plot_final)]
    fig, ax = plot_corr_heatmap(
        longdf_with_all,
        features=features_to_plot_final,
        session_order=["1","2","3","4","ALL"],
        annotate=True,
        by="session"
    )
    
    plt.show()
    
    
def _pick_contrast_col(df: pd.DataFrame) -> str:
    prefer = ["contrast", "improved_minus_declined", "Δmean", "delta", "diff"]
    numerics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    for name in prefer:
        if name in df.columns and pd.api.types.is_numeric_dtype(df[name]):
            return name
    if not numerics:
        raise ValueError("No numeric contrast column found.")
    return numerics[0]

def _top_k_pos_neg(df: pd.DataFrame, k: int):
    d = df.copy()
    col = _pick_contrast_col(d)
    d_pos = d.sort_values(col, ascending=False).head(k).copy()
    d_neg = d.sort_values(col, ascending=True ).head(k).copy()
    for c in d.select_dtypes(include="number").columns:
        d_pos[c] = d_pos[c].round(3)
        d_neg[c] = d_neg[c].round(3)
    return col, d_pos, d_neg

def _labelify_transition_pair(a: int, b: int) -> str:
    return f"{a}to{b}"

# ---------- main writer ----------
def write_fixed_transition_latex_with_meta(
    res_fixed_any: dict,
    out_tex: str = "fixed_transition_corr_tables.tex",
    note_excluded: str = "fixed pairs; excluding dangerous_weighted_distance",
    top_k_meta: int = 10,
    abs_thresh_single: float = None,   # filter in single-transition & ALL-transitions meta (e.g., 0.30)
    # pairwise thresholds
    abs_thresh_pair: float = 0.30,     # strong in BOTH transitions
    diff_thresh_pair: float = 0.40,    # same sign but large |Δr|
    top_k_pair: int = 10,
    # top-k improved vs declined
    top_k_contrast: int = 10,
    strength_thresh: int = 0.4
):
    """
    Works directly off res_fixed_any['delta_df'] produced by raw_delta_descriptives_flexible(..., pair_mode='fixed').
    Creates:
      1) Per-transition correlation tables (Δfeature vs Δperformance)
      2) ALL-transitions meta tables: Top shared (min|r|), Top dissimilar (range of r)
      3) Pairwise transition comparisons: Shared & Dissimilar (strong in both)
      4) Per-transition Top-k improved vs declined (from group means of Δ_* by perf_dir)
    """

    delta_df = res_fixed_any["delta_df"].copy()
    if delta_df.empty:
        with open(out_tex, "w") as f:
            f.write("% No delta rows in res_fixed_any['delta_df']\n")
        print(f"Wrote LaTeX (empty): {out_tex}")
        return

    # ensure perf_dir exists (should already from your function)
    if "perf_dir" not in delta_df.columns:
        def _label(x, eps=1e-9):
            if x > eps: return "improved"
            if x < -eps: return "declined"
            return "flat"
        delta_df["perf_dir"] = delta_df["Δ_performance"].apply(_label)

    # Define transition key strings like "0→1", "1→2", ...
    delta_df["transition_pair"] = delta_df.apply(
        lambda r: f"{int(r['session_earlier'])}→{int(r['session_later'])}", axis=1
    )

    # ---------- 1) Per-transition correlations ----------
    corr_by_transition: Dict[str, pd.DataFrame] = {}

    for tp, sub in delta_df.groupby("transition_pair"):
        if sub.empty or "Δ_performance" not in sub.columns:
            corr_by_transition[tp] = pd.DataFrame(columns=["feature","r","p","n"])
            continue

        # candidate Δ-features (exclude the target itself)
        dcols = [c for c in sub.columns if c.startswith("Δ_") and c != "Δ_performance"]
        rows: List[dict] = []

        # compute Spearman r, p, n per feature robustly
        for feat in dcols:
            xy = sub[[feat, "Δ_performance"]].dropna()
            # need variation and minimum rows
            if xy.shape[0] < 3 or xy[feat].nunique() <= 1 or xy["Δ_performance"].nunique() <= 1:
                continue
            r, p = spearmanr(xy[feat], xy["Δ_performance"])
            rows.append({"feature": feat, "r": float(r), "p": float(p), "n": int(len(xy))})

        df_stats = pd.DataFrame(rows)
        if not df_stats.empty:
            if abs_thresh_single is not None:
                df_stats = df_stats.assign(abs_r=df_stats["r"].abs())
                df_stats = df_stats[df_stats["abs_r"] >= abs_thresh_single].drop(columns="abs_r")
            df_stats = df_stats.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

        corr_by_transition[tp] = df_stats if not df_stats.empty else pd.DataFrame(columns=["feature","r","p","n"])
    # write per-transition LaTeX as before (but now columns are feature,r,p,n)
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with $\\Delta$performance by fixed session transitions\n\n")
        for tp, tbl in corr_by_transition.items():
            df = tbl.copy()
            if df.empty:
                continue
            if abs_thresh_single is not None:
                df = df.assign(abs_r=df["r"].abs())
                df = df[df["abs_r"] >= abs_thresh_single].drop(columns="abs_r")
            df = df.sort_values("r", key=lambda s: s.abs(), ascending=False)

            a, b = map(int, tp.split("→"))
            f.write(f"%% -------- Transition {tp} --------\n")
            f.write(df[["feature","r","p","n"]].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=f"Fixed session transition {tp}: Spearman correlations with $\\Delta$performance ({note_excluded})",
                label=f"tab:fixed_trans_corr_{_labelify_transition_pair(a, b)}"
            ))
            f.write("\n\n")

    # ---------- 2) ALL-transitions meta (shared/dissimilar across transitions) ----------
    long_rows = []
    for tp, tbl in corr_by_transition.items():
        df = tbl.copy()
        if df.empty: 
            continue
        col = _detect_corr_col_trans(df)
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df[col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({"transition": tp, "feature": getattr(r, "feature"), "corr": getattr(r, col)})

    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    meta_df = longdf.copy()
    if abs_thresh_single is not None:
        meta_df = meta_df[meta_df["abs_corr"] >= abs_thresh_single]

    agg = (meta_df.groupby("feature")
                 .agg(n_trans=("transition","nunique"),
                      mean_corr=("corr","mean"),
                      mean_abs=("abs_corr","mean"),
                      min_abs=("abs_corr","min"),
                      max_abs=("abs_corr","max"),
                      min_corr=("corr","min"),
                      max_corr=("corr","max"))
                 .reset_index())

    shared_all = agg.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k_meta).copy()
    agg["range_corr"] = agg["max_corr"] - agg["min_corr"]
    dissim_all = agg.sort_values("range_corr", ascending=False).head(top_k_meta).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    with open(out_tex, "a") as f:
        f.write("%% -------- ALL TRANSITIONS: Top shared features --------\n")
        f.write(shared_all[["feature","n_trans","min_abs","mean_abs","max_abs","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} features shared across fixed transitions "
                     f"(ranked by minimum |r| across transitions; {note_excluded})."),
            label="tab:fixed_trans_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- ALL TRANSITIONS: Top dissimilar features --------\n")
        f.write(dissim_all[["feature","n_trans","range_corr","min_corr","max_corr"]].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k_meta} most dissimilar features across fixed transitions "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:fixed_trans_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 3) Pairwise transition comparisons WITH p_diff and q_diff ----------
    transitions = sorted(corr_by_transition.keys())

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN FIXED TRANSITIONS (with p_diff & q_diff) --------\n\n")

    # Prepare per-transition dicts keyed by feature for join
    per_trans = {}
    for T in transitions:
        dfT = corr_by_transition[T]
        if dfT is None or dfT.empty:
            continue
        # Keep only strong enough features if you want to prefilter before testing
        dfT = dfT.copy()
        dfT["abs_r"] = dfT["r"].abs()
        if abs_thresh_pair is not None:
            dfT = dfT[dfT["abs_r"] >= abs_thresh_pair]
        per_trans[T] = dfT.set_index("feature")[["r","p","n","abs_r"]]

    # Build one big table of all pairwise tests
    pair_rows = []
    for A, B in itertools.combinations(transitions, 2):
        if A not in per_trans or B not in per_trans:
            continue
        A_df, B_df = per_trans[A], per_trans[B]
        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        if inter.empty:
            continue

        # require strong in BOTH transitions (AND), then apply an additional strength gate if desired
        inter = inter[(inter["abs_r_A"] >= strength_thresh) & (inter["abs_r_B"] >= strength_thresh)]
        if inter.empty:
            continue

        # compute Fisher p_diff per feature (guard min_n)
        pdiffs = []
        zvals = []
        for feat, r in inter.iterrows():
            rA, nA = r["r_A"], r["n_A"]
            rB, nB = r["r_B"], r["n_B"]
            if pd.isna(rA) or pd.isna(rB) or pd.isna(nA) or pd.isna(nB) or int(nA) < 4 or int(nB) < 4:
                z_diff, p_diff = (np.nan, np.nan)
            else:
                z_diff, p_diff = fisher_pdiff(float(rA), int(nA), float(rB), int(nB))
            zvals.append(z_diff); pdiffs.append(p_diff)

        inter = inter.assign(
            delta_r = inter["r_B"] - inter["r_A"],
            z_diff  = zvals,
            p_diff  = pdiffs,
            same_sign = np.sign(inter["r_A"]) == np.sign(inter["r_B"]),
            abs_delta_r = (inter["r_B"] - inter["r_A"]).abs(),
            min_abs = np.minimum(inter["abs_r_A"], inter["abs_r_B"])
        ).reset_index().rename(columns={"index":"feature"})

        # FDR within-pair (Benjamini–Hochberg) on available p_diff
        mask = inter["p_diff"].notna()
        if mask.any():
            _, q, _, _ = multipletests(inter.loc[mask, "p_diff"], method="fdr_bh")
            inter.loc[mask, "q_diff"] = q
        else:
            inter["q_diff"] = np.nan

        # SHARED (same sign), ranked by small |Δr| then min|r|
        shared = inter[inter["same_sign"]].sort_values(["abs_delta_r","min_abs"], ascending=[True, False]).head(top_k_pair)
        # DISSIMILAR (opposite sign or large |Δr|), ranked by |Δr| desc then min|r|
        dissim_mask = (inter["abs_delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(["abs_delta_r","min_abs"], ascending=[False, False]).head(top_k_pair)

        # write LaTeX
        a1, b1 = map(int, A.split("→")); a2, b2 = map(int, B.split("→"))
        cols = ["feature","r_A","p_A","n_A","r_B","p_B","n_B","delta_r","z_diff","p_diff","q_diff"]

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign) --------\n")
            f.write(shared[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Fixed transitions {A} vs {B}: shared-direction correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:fixed_trans_pair_shared_{_labelify_transition_pair(a1,b1)}_vs_{_labelify_transition_pair(a2,b2)}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Fixed transitions {A} vs {B}: dissimilar correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:fixed_trans_pair_dissimilar_{_labelify_transition_pair(a1,b1)}_vs_{_labelify_transition_pair(a2,b2)}"
            ))
            f.write("\n\n")

    # ---------- 4) Per-transition Top-k improved vs declined ----------
    # Compute means by perf_dir per transition, then contrast = improved − declined
    with open(out_tex, "a") as f:
        f.write("%% -------- PER-TRANSITION: Top-k improved vs declined (Δfeature means) --------\n\n")

    for tp, sub in delta_df.groupby("transition_pair"):
        dcols = [c for c in sub.columns if c.startswith("Δ_")]
        if not dcols or "perf_dir" not in sub.columns:
            continue
        agg = (sub.groupby("perf_dir")[dcols].mean().reset_index())
        labels = set(agg["perf_dir"].astype(str))
        if not {"improved","declined"}.issubset(labels):
            continue
        imp = agg[agg["perf_dir"]=="improved"].drop(columns=["perf_dir"], errors="ignore")
        dec = agg[agg["perf_dir"]=="declined"].drop(columns=["perf_dir"], errors="ignore")
        common = [c for c in imp.columns if c in dec.columns]
        if not common:
            continue
        contrast_df = pd.DataFrame((imp[common].values - dec[common].values), columns=common)
        contrast_df.insert(0, "contrast", "improved_minus_declined")

        col, top_impr, top_decl = _top_k_pos_neg(contrast_df, top_k_contrast)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- Transition {tp}: Top-k improved vs declined --------\n")
            f.write(top_impr.to_latex(
                index=False, float_format="%.3f",
                caption=f"Transition {tp}: top {top_k_contrast} factors when performance improved (highest positive {col}).",
                label=f"tab:fixed_trans_{_labelify_transition_pair(*map(int, tp.split('→')))}_topk_improved"
            ))
            f.write("\n\n")
            f.write(top_decl.to_latex(
                index=False, float_format="%.3f",
                caption=f"Transition {tp}: top {top_k_contrast} factors when performance declined (most negative {col}).",
                label=f"tab:fixed_trans_{_labelify_transition_pair(*map(int, tp.split('→')))}_topk_declined"
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (fixed transitions: per-transition, meta, pairwise, top-k contrasts) to: {out_tex}")

def build_features(df_sessions):
    """
    Reuses your existing `session_summary(row)` to create a tidy feature table.
    Requires columns: team_id, session, leader_type, performance (+ dict fields).
    """
    if 'leader_type' not in df_sessions.columns:
        raise ValueError("df_sessions must include 'leader_type'.")
    df = df_sessions.copy()
    df['leader_type'] = df['leader_type'].astype(str).str.lower().map(
        lambda s: 'human' if 'human' in s else ('ai' if 'ai' in s else s)
    )
    feats = df.apply(session_summary, axis=1)  # <-- your function from before
    out = pd.concat([df[['team_id','session','leader_type','performance','layout','suggestions',"name"]].reset_index(drop=True),
                     feats.reset_index(drop=True)], axis=1)
    return out
# -----------------------------------------------------------------------------


def summarize_up_vs_down(delta_df, by_leadership=True, same_leadership_only=False):
    """
    Produces tables of average Δfeatures for Δperformance>0 vs Δperformance<0.
    If by_leadership=True, returns dict with 'human' and 'ai' tables; else returns one table.
    """
    df = delta_df.copy()
    df = df.dropna(subset=['Δ_performance'])
    # label improves / declines
    df['perf_dir'] = np.where(df['Δ_performance']>0, 'improved',
                        np.where(df['Δ_performance']<0, 'declined', 'flat'))
    # we’ll ignore exact zeros to make contrasts cleaner
    df = df[df['perf_dir']!='flat']

    delta_cols = [c for c in df.columns if c.startswith('Δ_')]

    def _agg(tbl):
        return (tbl
                .groupby('perf_dir')[delta_cols]
                .mean()
                .rename_axis(None)
                .reset_index())

    if by_leadership:
        out = {}
        for lt in ['human','ai']:
            sub = df[df['leader_type']==lt]
            if sub.empty:
                out[lt] = pd.DataFrame(columns=['perf_dir']+delta_cols)
            else:
                out[lt] = _agg(sub)
        return out
    else:
        return _agg(df)


def diff_improved_minus_declined(summary_tbl):
    """
    Convenience: turn the 'improved vs declined' table into a single-row
    'improved - declined' contrast for easier reading per leadership.
    """
    if summary_tbl.empty or 'perf_dir' not in summary_tbl.columns:
        return pd.DataFrame()
    if not set(['improved','declined']).issubset(set(summary_tbl['perf_dir'])):
        return pd.DataFrame()
    imp = summary_tbl[summary_tbl['perf_dir']=='improved'].drop(columns=['perf_dir']).copy()
    dec = summary_tbl[summary_tbl['perf_dir']=='declined'].drop(columns=['perf_dir']).copy()
    contrast = (imp.values - dec.values)
    out = pd.DataFrame(contrast, columns=imp.columns)
    out.insert(0, 'contrast', ['improved_minus_declined'])
    return out


def correlations_with_dperf(delta_df, by_leadership=True):
    """
    Simple Pearson correlations between Δperformance and each Δfeature.
    Returns one table (overall) or dict by leadership type.
    """
    df = delta_df.dropna(subset=['Δ_performance']).copy()
    delta_cols = [c for c in df.columns if c.startswith('Δ_')]

    def _corr_tbl(tbl):
        if tbl.empty: 
            return pd.DataFrame(columns=['feature','corr_with_Δperf'])
        corrs = tbl[delta_cols].corr(numeric_only=True, method='spearman')['Δ_performance'].dropna().sort_values(ascending=False)
        return corrs.rename('corr_with_Δperf').reset_index().rename(columns={'index':'feature'})

    if by_leadership:
        out = {}
        for lt in ['human','ai']:
            sub = df[df['leader_type']==lt]
            out[lt] = _corr_tbl(sub)
        return out
    else:
        return _corr_tbl(df)


def _delta_between(later_row: pd.Series, earlier_row: pd.Series, id_cols=('team_id','session','leader_type')):
    """
    Compute Δ for all numeric columns between two rows (later - earlier).
    Returns a single-row DataFrame with id columns taken from 'later' plus Δ_* columns.
    """
    later = later_row.copy()
    earlier = earlier_row.copy()
    # Identify numeric columns to diff (exclude ids)
    num_cols = [c for c in later.index if c not in id_cols and pd.api.types.is_numeric_dtype(type(later[c]))]
    out = {k: later[k] for k in id_cols}
    for c in num_cols:
        try:
            out[f"Δ_{c}"] = float(later[c]) - float(earlier[c])
        except Exception:
            out[f"Δ_{c}"] = np.nan
    return pd.DataFrame([out])


def build_deltas_prev(feat_df: pd.DataFrame, same_leadership_only: bool = False) -> pd.DataFrame:
    """
    Non-contiguous friendly 'previous' deltas:
    For each (team[, leader]) row, compute difference against the MOST RECENT PRIOR row
    in that group (by session). If none exists, Δ_* are NaN for that row.

    same_leadership_only=False: group by team only.
    same_leadership_only=True : group by (team, leader_type).
    """
    keys = ['team_id'] if not same_leadership_only else ['team_id','leader_type']
    df = feat_df.sort_values(keys + ['session']).copy()

    # shift within group gives you the most recent prior matching those keys (non-contiguous globally is fine)
    grp = df.groupby(keys, group_keys=False)
    prev = grp.shift(1)

    # Build deltas row-wise
    delta_rows = []
    for idx in df.index:
        later_row = df.loc[idx]
        earlier_row = prev.loc[idx]
        if earlier_row.isna().all():
            # No previous in this group
            # Still return row with ids and Δ_performance (and others) = NaN
            delta_rows.append(_delta_between(later_row, later_row).assign(**{k: np.nan for k in list(_ for _ in _delta_between(later_row, later_row).columns if _.startswith('Δ_'))}).iloc[[0]])
        else:
            delta_rows.append(_delta_between(later_row, earlier_row).assign(gap_sessions = later_row['session'] - earlier_row['session']).iloc[[0]])

    delta_df = pd.concat(delta_rows, ignore_index=True)
    # keep id cols + Δ_* (and gap if present)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


def build_deltas_all_pairs(feat_df: pd.DataFrame, same_leadership_only: bool = False) -> pd.DataFrame:
    """
    Exhaustive all-pairs deltas:
    For each team (and optionally within each leader type), compute differences for ALL pairs
    (earlier -> later), regardless of contiguity. Adds gap_sessions = later - earlier.

    Warning: grows O(n^2) per group; fine for 4 sessions but be mindful if larger.
    """
    keys = ['team_id'] if not same_leadership_only else ['team_id','leader_type']
    rows = []
    for _, g in feat_df.sort_values(keys + ['session']).groupby(keys):
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for i, j in combinations(range(len(g)), 2):
            earlier = g.loc[i]
            later   = g.loc[j]
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
            rows.append(d)
    if not rows:
        # empty
        cols = ['team_id','session','leader_type','gap_sessions']
        return pd.DataFrame(columns=cols)
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


def raw_delta_descriptives_flexible(df_sessions: pd.DataFrame,
                                    pair_mode: str = "prev",
                                    same_leadership_only: bool = False,
                                    keep_flats: bool = False,
                                    eps: float = 1e-9):
    """
    Flexible wrapper:
      - pair_mode: "prev" for most-recent prior deltas (non-contiguous OK),
                   "all_pairs" for exhaustive all-pair deltas within group.
      - same_leadership_only: if True, compare only within same leader type.
      - keep_flats: if True, keep Δ_performance ~ 0 rows; else drop to sharpen contrasts.
    """
    feat_df = df_sessions.copy()

    # ---------------------------
    # Build delta_df by pair_mode
    # ---------------------------
    if pair_mode == "prev":
        # existing behavior: previous-within team or team+leader
        delta_df = build_deltas_prev(feat_df, same_leadership_only=same_leadership_only)

    elif pair_mode == "all_pairs":
        # existing behavior: all pairs within team or team+leader
        delta_df = build_deltas_all_pairs(feat_df, same_leadership_only=same_leadership_only)

    elif pair_mode == "fixed":
        # NEW: only (0→1), (1→2), (2→3), (0→3) for each team
        rows = []
        # we will optionally check same-leadership after picking pairs
        wanted_pairs = [(0,1), (1,2), (2,3), (0,3)]
        for tid, g in feat_df.groupby("team_id"):
            g = g.sort_values("session").reset_index(drop=True)
            # build dict session_index -> row for quick access
            sess_to_row = {int(g.loc[i, "session"]): g.loc[i] for i in g.index}
            for i, j in wanted_pairs:
                if i in sess_to_row and j in sess_to_row:
                    earlier = sess_to_row[i]
                    later   = sess_to_row[j]
                    # filter by same leadership if requested
                    if same_leadership_only and (earlier["leader_type"] != later["leader_type"]):
                        continue
                    # compute deltas
                    num_cols = [c for c in feat_df.columns if c not in ["team_id","session","leader_type",'layout','suggestions',"name"]]
                    out = {
                        "team_id": earlier["team_id"],
                        "leader_type": later["leader_type"],  # leader at later session (for summaries)
                        "session_earlier": int(earlier["session"]),
                        "session_later": int(later["session"]),
                        "gap_sessions": int(later["session"] - earlier["session"]),
                    }
                    for c in num_cols:
                        try:
                            out[f"Δ_{c}"] = (later[c] - earlier[c]) if pd.notna(later[c]) and pd.notna(earlier[c]) else np.nan
                        except:
                            pdb.set_trace()
                    rows.append(out)
        delta_df = pd.DataFrame(rows)

    else:
        raise ValueError("pair_mode must be 'prev', 'all_pairs', or 'fixed'")

    # ----------------------------------
    # Label direction and (optionally) drop flats
    # ----------------------------------
    if 'Δ_performance' in delta_df.columns:
        def _label(x):
            if x > eps: return 'improved'
            if x < -eps: return 'declined'
            return 'flat'
        delta_df['perf_dir'] = delta_df['Δ_performance'].apply(_label)
        if not keep_flats:
            delta_df = delta_df[delta_df['perf_dir']!='flat'].copy()


    # ---------------------------
    # Build summaries per leader (robust)
    # ---------------------------
    def _agg(tbl):
        """Average of Δ_* by perf_dir; always returns a DataFrame with a 'perf_dir' column (may be empty)."""
        if tbl.empty or 'perf_dir' not in tbl.columns:
            return pd.DataFrame(columns=['perf_dir'])
        delta_cols = [c for c in tbl.columns if c.startswith('Δ_')]
        if not delta_cols:
            return pd.DataFrame(columns=['perf_dir'])
        out = (tbl.groupby('perf_dir')[delta_cols]
                  .mean()
                  .reset_index())
        # Ensure column present even if no groups materialize
        if 'perf_dir' not in out.columns:
            out.insert(0, 'perf_dir', pd.Series(dtype=object))
        return out

    updown_by_leader, contrast_by_leader, corr_by_leader = {}, {}, {}

    for lt in ['human','ai']:
        sub = delta_df[delta_df['leader_type'] == lt].copy()
        up = _agg(sub)
        updown_by_leader[lt] = up

        # Contrast improved - declined (guarded)
        if (not up.empty) and ('perf_dir' in up.columns):
            labels = set(up['perf_dir'].astype(str).tolist())
            if {'improved', 'declined'}.issubset(labels):
                imp = up[up['perf_dir'] == 'improved'].drop(columns=['perf_dir'], errors='ignore')
                dec = up[up['perf_dir'] == 'declined'].drop(columns=['perf_dir'], errors='ignore')
                common = [c for c in imp.columns if c in dec.columns]
                con_df = (pd.DataFrame((imp[common].values - dec[common].values), columns=common)
                          if common else pd.DataFrame())
                if not con_df.empty:
                    con_df.insert(0, 'contrast', 'improved_minus_declined')
            else:
                con_df = pd.DataFrame()
        else:
            con_df = pd.DataFrame()
        contrast_by_leader[lt] = con_df

        # Correlations with Δperformance (guarded)
        if (not sub.empty) and ('Δ_performance' in sub.columns):
            dcols = [c for c in sub.columns if c.startswith('Δ_')]
            if dcols:
                corr_col = sub[dcols].corr(numeric_only=True, method='spearman').get('Δ_performance')
                if corr_col is not None:
                    corrs = corr_col.dropna().sort_values(ascending=False)
                    corr_tbl = (corrs.rename('corr_with_Δperf')
                                     .reset_index()
                                     .rename(columns={'index': 'feature'}))
                else:
                    corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
            else:
                corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
        else:
            corr_tbl = pd.DataFrame(columns=['feature','corr_with_Δperf'])
        corr_by_leader[lt] = corr_tbl

    return {
        'feat_df': feat_df,
        'delta_df': delta_df,
        'updown_by_leader': updown_by_leader,
        'contrast_by_leader': contrast_by_leader,
        'corr_by_leader': corr_by_leader
    }
    



def build_deltas_prev_samelead(feat_df: pd.DataFrame) -> pd.DataFrame:
    keys = ['team_id','leader_type']
    df = feat_df.sort_values(keys + ['session']).copy()
    grp = df.groupby(keys, group_keys=False)
    prev = grp.shift(1)
    rows = []
    for idx in df.index:
        later = df.loc[idx]; earlier = prev.loc[idx]
        if earlier.isna().all():
            d = _delta_between(later, later);  # scaffold ids
            for c in list(d.columns):
                if c.startswith('Δ_'): d[c] = np.nan
        else:
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
        rows.append(d)
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]

def build_deltas_allpairs_samelead(feat_df: pd.DataFrame) -> pd.DataFrame:
    keys = ['team_id','leader_type']
    rows = []
    for _, g in feat_df.sort_values(keys + ['session']).groupby(keys):
        g = g.reset_index(drop=True)
        for i, j in combinations(range(len(g)), 2):
            earlier, later = g.loc[i], g.loc[j]
            d = _delta_between(later, earlier)
            d['gap_sessions'] = later['session'] - earlier['session']
            rows.append(d)
    if not rows:
        return pd.DataFrame(columns=['team_id','session','leader_type','gap_sessions'])
    delta_df = pd.concat(rows, ignore_index=True)
    keep = [c for c in delta_df.columns if c in ['team_id','session','leader_type','gap_sessions'] or c.startswith('Δ_')]
    return delta_df[keep]


SIGN_HELP_BY_LEADER = {
    "human": {
        "Δ_precision": +1, "Δ_benign_ratio": -1,
        "Δ_danger_weight_coverage": +1, "Δ_dangerous_weighted_distance": +1,
        "Δ_median_sens_danger_carried": +1, "Δ_pct_danger_carried_ge2": +1,
        "Δ_misunderstanding_rate": -1, "Δ_change_rate": -1,
        "Δ_completion_rate": +1, "Δ_confirmation_rate": +1,
        "Δ_latency_norm_median": -1, "Δ_path_eff_median": -1, "Δ_mobility_gini": -1,
        "Δ_incomplete_order_rate": -1, "Δ_underspecified_rate": -1,
        # sensing mix: human-specific
        "Δ_sensed_2": -1, "Δ_sensed_3": +1,
        "Δ_sensed_4": 0, "Δ_sensed_1": 0, "Δ_sensed_0": 0,
        # leave survey/UX items neutral (0) unless you have theory
    },
    "ai": {
        "Δ_precision": +1, "Δ_benign_ratio": -1,
        "Δ_danger_weight_coverage": +1, "Δ_dangerous_weighted_distance": +1,
        "Δ_median_sens_danger_carried": +1, "Δ_pct_danger_carried_ge2": +1,
        "Δ_misunderstanding_rate": -1, "Δ_change_rate": -1,
        "Δ_completion_rate": +1, "Δ_confirmation_rate": +1,
        "Δ_latency_norm_median": -1, "Δ_path_eff_median": -1, "Δ_mobility_gini": -1,
        "Δ_incomplete_order_rate": -1, "Δ_underspecified_rate": -1,
        # sensing mix: keep neutral (low/unstable effects in report)
        "Δ_sensed_2": 0, "Δ_sensed_3": 0, "Δ_sensed_4": 0, "Δ_sensed_1": 0, "Δ_sensed_0": 0,
    }
}

# Optionally zero out tiny-importance features (absolute corr < threshold)
MIN_ABS_CORR = 0.15  # tune if you like

def importance_from_correlations(delta_df: pd.DataFrame, leader_type: str) -> dict:
    sub = delta_df[(delta_df['leader_type']==leader_type)].dropna(subset=['Δ_performance']).copy()
    if sub.empty:
        return {}
    delta_cols = [c for c in sub.columns if c.startswith('Δ_')]
    corrs = sub[delta_cols].corr(numeric_only=True, method='spearman')['Δ_performance'].dropna()
    # keep only signals above threshold; otherwise weight=0 (ignored)
    imp = {k: (abs(v) if abs(v) >= MIN_ABS_CORR else 0.0) for k,v in corrs.items()}
    return imp

def explain_transition(delta_row: pd.Series, importance: dict, top_k: int = 3, leader_type: str = None):
    items = []
    for feat, val in delta_row.items():
        if not (feat.startswith('Δ_') and feat != 'Δ_performance'): 
            continue
        w = importance.get(feat, 0.0)
        score = abs(val) * w
        if np.isfinite(val) and np.isfinite(score):
            items.append((feat, float(val), score))
    items.sort(key=lambda x: x[2], reverse=True)
    top = items[:max(top_k, 0)]

    # leader-specific sign map
    sign_map = SIGN_HELP_BY_LEADER.get(leader_type, {})
    dperf = float(delta_row.get('Δ_performance', np.nan))
    improving, worsening = [], []
    for f, v, s in top:
        pref = sign_map.get(f, 0)
        if pref == 0:
            # fall back: align with Δperformance sign
            helped = (np.sign(v) == np.sign(dperf))
        else:
            helped = (np.sign(v) == np.sign(pref))
        (improving if helped else worsening).append(f"{f} ({v:+.3f})")

    return {
        "top_contributors": [f"{f} ({v:+.3f})" for f,v,_ in top],
        "top_improving": improving,
        "top_worsening": worsening
    }

# ---------- Orchestrator: build explanations for SAME LEADERSHIP only ----------
def per_team_explanations_same_leadership(df_sessions: pd.DataFrame,
                                          pair_mode: str = "prev",
                                          top_k: int = 3,
                                          use_sign_map: bool = True):
    """
    pair_mode: "prev" (most recent prior within same leader) OR "all_pairs" (every earlier→later same leader)
    Returns a tidy DataFrame where each row is a same-leadership transition with explanation fields.
    """
    feat_df = df_sessions.copy()
    

    if pair_mode == "prev":
        delta_df = build_deltas_prev_samelead(feat_df)
    elif pair_mode == "all_pairs":
        delta_df = build_deltas_allpairs_samelead(feat_df)
    else:
        raise ValueError("pair_mode must be 'prev' or 'all_pairs'")

    # Drop rows with no Δ_performance
    delta_df = delta_df.dropna(subset=['Δ_performance']).copy()

    # Build importance per leadership
    importances = {lt: importance_from_correlations(delta_df, lt) for lt in ['human','ai']}

    # Build explanations per row
    rows = []
    for _, r in delta_df.iterrows():
        lt = r['leader_type']
        imp = importances.get(lt, {})
        expl = explain_transition(r, imp, top_k=top_k, leader_type=lt)
        
        rows.append({
            "team_id": r['team_id'],
            "leader_type": lt,
            "session_later": r['session'],
            "gap_sessions": r.get('gap_sessions', np.nan),
            "Δ_performance": r['Δ_performance'],
            "top_contributors": "; ".join(expl["top_contributors"]),
            "top_improving": "; ".join(expl["top_improving"]),
            "top_worsening": "; ".join(expl["top_worsening"]),
        })
    return pd.DataFrame(rows)

# ---------- Minimal print helper ----------
def print_team_explanations_samelead(df_sessions, pair_mode="prev", top_k=3, export_csv=None):
    tbl = per_team_explanations_same_leadership(df_sessions, pair_mode=pair_mode, top_k=top_k)
    # Show all columns fully
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)

    print(tbl.sort_values(['leader_type','team_id','session_later']).reset_index(drop=True))
    if export_csv:
        tbl.to_csv(export_csv, index=False)
        print(f"\nSaved to {export_csv}")

def correlations_per_layout_no_transitions(
    df_sessions: pd.DataFrame,
    layout_col: str = 'layout',
    exclude_patterns = (
        r"^dangerous_?weighted_?distance$",
        r"^dangerous_?weithed_?distance$",  # common typo
    ),
    abs_threshold: float = None,  # e.g., 0.3 to keep only |r|>0.3
    min_rows: int = 2
):
    """
    Session-level (no deltas). For each layout, correlate performance with every numeric feature.
    Returns: {'feat_df': DataFrame, 'corr_by_layout': {layout_value: DataFrame(feature, corr_with_performance)}}
    """
    feat_df = df_sessions.copy()
    if layout_col not in feat_df.columns:
        raise ValueError(f"Layout column '{layout_col}' not found in features table.")

    # Ensure performance is numeric
    if not pd.api.types.is_numeric_dtype(feat_df['performance']):
        feat_df['performance'] = pd.to_numeric(feat_df['performance'], errors='coerce')

    # Collect numeric features (exclude id-like cols)
    id_like = {'team_id','session','leader_type', layout_col}
    numeric_feats = [c for c in feat_df.columns
                     if c not in id_like and pd.api.types.is_numeric_dtype(feat_df[c])]

    # Exclude tautological features
    import re
    excl = re.compile("|".join(exclude_patterns), flags=re.IGNORECASE)
    numeric_feats = [c for c in numeric_feats if not excl.search(c)]
    corr_by_layout = {}
    for lv, sub in feat_df.groupby(layout_col):
        sub = sub.copy()
        if len(sub) < min_rows:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Keep features that exist and are numeric in this subset AND have variance
        cols_here = []
        for c in numeric_feats:
            if c in sub.columns and pd.api.types.is_numeric_dtype(sub[c]):
                # drop zero-variance to avoid NaNs in corr
                if sub[c].nunique(dropna=True) > 1:
                    cols_here.append(c)

        if not cols_here or sub['performance'].nunique(dropna=True) <= 1:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Build correlation matrix explicitly on [performance] + features
        cols = ['performance'] + cols_here
        mat = sub[cols].corr(numeric_only=True, method='spearman')

        # Guard against degenerate cases
        if 'performance' not in mat.columns or 'performance' not in mat.index:
            corr_by_layout[lv] = pd.DataFrame(columns=['feature','corr_with_performance'])
            continue

        # Get the single 'performance' column as a 1D Series (not a DataFrame)
        corr_s = mat.loc[cols_here, ['performance']].iloc[:, 0]  # <-- Series
        corr_s.name = 'corr_with_performance'

        # Now build a clean two-column table
        tbl = (corr_s.dropna()
                      .reset_index()
                      .rename(columns={'index': 'feature'}))

        # Rank by |r|
        tbl['abs_corr'] = tbl['corr_with_performance'].abs()
        tbl = tbl.sort_values('abs_corr', ascending=False).drop(columns=['abs_corr']).reset_index(drop=True)

        if abs_threshold is not None:
            tbl = tbl[tbl['corr_with_performance'].abs() > abs_threshold].reset_index(drop=True)

        corr_by_layout[lv] = tbl

    return {'feat_df': feat_df, 'corr_by_layout': corr_by_layout}


def write_per_layout_latex(
    corr_by_layout: dict,
    out_tex: str = "layout_session_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance"
):
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: session-level correlations with performance per layout\n\n")
        for lv, tbl in corr_by_layout.items():
            f.write(f"%% -------- Layout {lv} --------\n")
            f.write(tbl.to_latex(index=False, float_format="%.3f",
                                 caption=f"Session-level correlations with performance for layout {lv} ({note_excluded})",
                                 label=f"tab:session_corr_layout_{str(lv).replace(' ','_')}"))
            f.write("\n\n")
    print(f"Wrote LaTeX tables to: {out_tex}")
    
def _detect_corr_col(df: pd.DataFrame) -> str:
    preferred = ["corr_with_Δperf", "corr_with_perf", "corr", "r"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found.")

def write_per_layout_latex_with_meta(
    corr_by_layout: Dict[str, pd.DataFrame],
    out_tex: str = "layout_session_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 10,
    abs_thresh: Optional[float] = None,       # e.g., 0.30 per-layout filter for the single-layout tables & meta tables
    require_all_layouts: bool = True,
    min_layouts: Optional[int] = None,
    # NEW: pairwise comparison knobs
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.0,            # feature must be |r|>=this in BOTH layouts to be considered in a pair
    diff_thresh_pair: float = 0.40,           # if same sign but |Δr|>=this, we still call it "dissimilar"
    top_k_pair: int = 50,
    pretty_layout_names: Optional[Dict[str, str]] = None,
    strength_thresh: int = 0.4
):
    """
    Adds pairwise comparisons between layouts:
      For each pair (A,B), we produce:
        - 'Shared' table: features strong in both (|r|>=abs_thresh_pair) AND same sign; ranked by min(|r_A|,|r_B|)
        - 'Dissimilar' table: features strong in both but opposite sign OR |r_A - r_B| >= diff_thresh_pair; ranked by |r_A - r_B|

    Everything else (per-layout tables and cross-layout shared/dissimilar meta tables) stays as before.
    """
    # ---------- 1) Write the original per-layout tables ----------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: session-level correlations with performance per layout\n\n")
        for lv, tbl in corr_by_layout.items():
            name = pretty_layout_names.get(lv, str(lv)) if pretty_layout_names else str(lv)
            df = tbl.copy()
            corr_col = _detect_corr_col(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Layout {name} --------\n")
            f.write(df.to_latex(index=False, float_format="%.3f",
                                caption=f"Session-level correlations with performance for layout {name} ({note_excluded})",
                                label=f"tab:session_corr_layout_{str(name).replace(' ','_')}"))
            f.write("\n\n")

    # ---------- 2) Build a long-form table across layouts (used for meta + pairwise) ----------
    long_rows = []
    for lv, tbl in corr_by_layout.items():
        name = pretty_layout_names.get(lv, str(lv)) if pretty_layout_names else str(lv)
        df = tbl.copy()
        corr_col = _detect_corr_col(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "layout": name,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
            })
    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return
    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()

    # ---------- 3) Cross-layout meta (shared/dissimilar across ALL layouts) ----------
    layouts = sorted(longdf["layout"].unique().tolist())
    nL = len(layouts)
    minL = nL if require_all_layouts else (min_layouts if min_layouts is not None else max(1, int(round(0.75*nL))))

    longdf_meta = longdf.copy()
    if abs_thresh is not None:
        longdf_meta = longdf_meta[longdf_meta["abs_corr"] >= abs_thresh]
    agg = (
        longdf_meta.groupby("feature")
                   .agg(n_layouts=("layout","nunique"),
                        mean_corr=("corr","mean"),
                        mean_abs=("abs_corr","mean"),
                        min_abs=("abs_corr","min"),
                        max_abs=("abs_corr","max"),
                        min_corr=("corr","min"),
                        max_corr=("corr","max"))
                   .reset_index()
    )
    kept = agg[agg["n_layouts"] >= minL].copy()
    if kept.empty:
        kept = agg.copy()

    shared_all = kept.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    kept["range_corr"] = kept["max_corr"] - kept["min_corr"]
    dissim_all = kept.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_layouts","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_layouts","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- CROSS-LAYOUT (ALL): Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across layouts (ranked by minimum |r| across layouts; {note_excluded})."),
            label="tab:layout_shared_top"
        ))
        f.write("\n\n")
        f.write("%% -------- CROSS-LAYOUT (ALL): Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across layouts (ranked by correlation range; {note_excluded})."),
            label="tab:layout_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 4) NEW: Pairwise comparisons between layouts ----------
    if not do_pairwise or len(layouts) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN LAYOUTS --------\n\n")

    # Convenience: pre-split per layout with abs-threshold applied for pairwise logic
    per_layout_df = {}
    for L in layouts:
        sub = longdf[longdf["layout"] == L].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_layout_df[L] = sub.set_index("feature")[["corr","abs_corr"]]

    # For each pair (A,B): compute shared & dissimilar sets
    for A, B in itertools.combinations(layouts, 2):
        A_df = per_layout_df[A]
        B_df = per_layout_df[B]
        if A_df.empty or B_df.empty:
            continue
        # intersection: features strong in BOTH
        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")
        
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | 
              (inter["abs_corr_B"] >= strength_thresh)]
        
        if inter.empty:
            continue

        # Determine sign
        inter["sign_A"] = inter["corr_A"].apply(lambda x: 1 if x >= 0 else -1)
        inter["sign_B"] = inter["corr_B"].apply(lambda x: 1 if x >= 0 else -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        # "Shared": strong in both, same sign; rank by min_abs
        shared = inter[inter["same_sign"]].sort_values(["delta_r", "min_abs"], ascending=[True, False]).head(top_k_pair).copy()

        # "Dissimilar": strong in both AND (opposite sign OR big magnitude gap)
        dissim_mask = (~inter["same_sign"]) | (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(["delta_r","min_abs"], ascending=[False, False]).head(top_k_pair).copy()

        # Select & round
        shared_tbl = shared.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r"]].copy()
        shared_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})", "min_abs":"min|r|", "delta_r":"|Δr|"}, inplace=True)
        dissim_tbl = dissim.reset_index()[["feature","corr_A","corr_B","min_abs","delta_r","same_sign"]].copy()
        dissim_tbl.rename(columns={"corr_A": f"r({A})", "corr_B": f"r({B})", "min_abs":"min|r|", "delta_r":"|Δr|", "same_sign":"same_sign"}, inplace=True)

        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        # Append to LaTeX
        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {A} vs {B}: shared strong correlations (|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:layout_pair_shared_{A}_vs_{B}".replace(" ","_")
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {A} vs {B}: dissimilar strong correlations (|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:layout_pair_dissimilar_{A}_vs_{B}".replace(" ","_")
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise comparisons) to: {out_tex}")
    
def plot_layout_heatmap(corr_by_layout: dict, top_k: int = 15, out_png: str = "layout_corr_heatmap.png"):
    # Stack into one DataFrame
    stacks = []
    for lv, tbl in corr_by_layout.items():
        if tbl.empty: 
            continue
        t = tbl.copy()
        t['layout'] = lv
        stacks.append(t)
    if not stacks:
        print("No data for heatmap.")
        return
    allc = pd.concat(stacks, ignore_index=True)

    # pick top features by max |corr| across layouts
    top_feats = (allc.assign(abs_corr=lambda d: d['corr_with_performance'].abs())
                      .groupby('feature')['abs_corr'].max()
                      .sort_values(ascending=False)
                      .head(top_k).index.tolist())

    pivot = (allc[allc['feature'].isin(top_feats)]
         .pivot_table(index='feature',
                      columns='layout',
                      values='corr_with_performance',
                      aggfunc='mean')  # or 'max', 'first'
         .fillna(0.0))

    pivot.index = pivot.index.to_series().apply(lambda s: str(s).split()[0].split('_')[0])

    # simple matplotlib heatmap (no seaborn)
    fig_w = 1.0 * max(8, len(pivot.columns) * 1.5)
    fig_h = 0.6 * max(6, len(pivot.index) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    data = pivot.values.astype(float)
    im = ax.imshow(data, aspect='auto', cmap='coolwarm', vmin=-1.0, vmax=1.0)

    # axis labels
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    ax.set_title("Session-level correlations with performance by layout (top {} features)".format(top_k))
    ax.set_xlabel("Layout")
    ax.set_ylabel("Feature")

    # annotate
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha='center', va='center', color='white' if abs(v) > 0.5 else 'black', fontsize=9)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation (r)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()
    print(f"Saved heatmap to: {out_png}")

def _to_binary_suggestions(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, np.integer)):
        return int(val) if val in (0,1) else np.nan
    if isinstance(val, float):
        return int(val) if val in (0.0, 1.0) else np.nan

    s = str(val).strip().lower()
    truthy = {"1","true","t","yes","y","on","with","enabled"}
    falsy  = {"0","false","f","no","n","off","without","disabled"}
    if s in truthy: return 1
    if s in falsy:  return 0

    # last-ditch attempt (e.g., "1.0")
    try:
        f = float(s)
        return int(f) if f in (0.0, 1.0) else np.nan
    except:
        return np.nan


def _detect_corr_col(df: pd.DataFrame) -> str:
    preferred = ["corr_with_Δperf", "corr_with_perf", "corr", "r"]
    for c in preferred:
        if c in df.columns:
            return c
    for c in df.columns:
        if c != "abs_corr" and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError("No numeric correlation column found in provided DataFrame.")

def _key_to_label(k: Hashable) -> str:
    if isinstance(k, tuple):
        return "_".join(str(x) for x in k)
    return str(k)

def _detect_cols(df: pd.DataFrame):
    # correlation
    corr_candidates = ["corr","r","rho","spearman_r","spearman_rho","corr_with_performance"]
    p_candidates    = ["p","p_value","pval","pval_two_sided"]
    n_candidates    = ["n","n_obs","n_samples","N"]
    def _pick(opts):
        for c in opts:
            if c in df.columns: return c
        raise KeyError(f"Could not find any of {opts} in columns: {list(df.columns)}")
    return _pick(corr_candidates), _pick(p_candidates), _pick(n_candidates)

def _bh_fdr(pvals: pd.Series) -> pd.Series:
    """Benjamini-Hochberg (non-negative pvals, NaNs preserved)."""
    s = pvals.copy()
    mask = s.notna()
    x = s[mask].values
    m = len(x)
    order = np.argsort(x)
    ranks = np.empty_like(order); ranks[order] = np.arange(1, m+1)
    q = x * m / ranks
    # ensure monotone non-increasing after sorting by p
    q_sorted = np.minimum.accumulate(q[order[::-1]])[::-1]
    # clip to [0,1]
    q_sorted = np.clip(q_sorted, 0, 1)
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[mask] = q_sorted
    return out




def write_by_leadership_suggestions_latex_with_meta(
    corr_by_ls: Dict[Hashable, pd.DataFrame],
    out_tex: str = "lead_sugg_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    top_k: int = 10,
    abs_thresh: Optional[float] = None,        # per-group filter for single-group & meta tables
    require_all_groups: bool = True,
    min_groups: Optional[int] = None,
    pretty_names: Optional[Dict[Hashable, str]] = None,
    # NEW: pairwise controls
    do_pairwise: bool = True,
    abs_thresh_pair: float = 0.40,             # strong in BOTH groups means |r|>=this in each
    diff_thresh_pair: float = 0.40,            # same sign but big magnitude gap counts as dissimilar if |Δr|>=this
    top_k_pair: int = 50,
    strength_thresh: int = 0.4,
    raw_df = None
):
    """
    Writes:
      • One LaTeX table per (leadership × suggestions) group (as before)
      • Cross-group meta: top shared (min |r|) and top dissimilar (range of r) across ALL groups
      • NEW Pairwise tables for every group pair:
          - Shared strong correlations (same sign; |r|>=abs_thresh_pair in both)
          - Dissimilar strong correlations (opposite sign OR |Δr|>=diff_thresh_pair; |r|>=abs_thresh_pair in both)
    """
    # ---------- 1) Per-group tables ----------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: correlations with performance by leadership × suggestions groups\n\n")
        for gk, tbl in corr_by_ls.items():
            label = pretty_names.get(gk, _key_to_label(gk)) if pretty_names else _key_to_label(gk)
            df = tbl.copy()
            corr_col = _detect_corr_col(df)
            df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
            df = df[df[corr_col].notna()].copy()
            if abs_thresh is not None:
                df = df.assign(abs_corr=df[corr_col].abs())
                df = df[df["abs_corr"] >= abs_thresh].drop(columns="abs_corr")
            f.write(f"%% -------- Group: {label} --------\n")
            f.write(df.to_latex(
                index=False, float_format="%.3f",
                caption=f"Correlations with performance — Group {label} ({note_excluded})",
                label=f"tab:lead_sugg_corr_{label.replace(' ','_').replace('/','-')}"
            ))
            f.write("\n\n")

    # ---------- 2) Long form across groups (keep r, p, n) ----------
    long_rows = []
    group_key_to_label = {}
    for gk, tbl in corr_by_ls.items():
        label = pretty_names.get(gk, _key_to_label(gk)) if pretty_names else _key_to_label(gk)
        group_key_to_label[gk] = label
        df = tbl.copy()
        corr_col, p_col, n_col = _detect_cols(df)
        df[corr_col] = pd.to_numeric(df[corr_col], errors="coerce")
        df[p_col]    = pd.to_numeric(df[p_col],    errors="coerce")
        df[n_col]    = pd.to_numeric(df[n_col],    errors="coerce")
        df = df[df[corr_col].notna()].copy()
        for r in df.itertuples(index=False):
            long_rows.append({
                "group": label,
                "feature": getattr(r, "feature"),
                "corr": getattr(r, corr_col),
                "p":    getattr(r, p_col),
                "n":    getattr(r, n_col),
            })

    if not long_rows:
        print(f"Wrote LaTeX tables to: {out_tex} (no rows)")
        return

    longdf = pd.DataFrame(long_rows)
    longdf["abs_corr"] = longdf["corr"].abs()
    groups = sorted(longdf["group"].unique().tolist())

    # ---------- 3) Cross-group meta (ALL groups) ----------
    longdf_meta = longdf.copy()
    if abs_thresh is not None:
        longdf_meta = longdf_meta[longdf_meta["abs_corr"] >= abs_thresh]
    nG = len(groups)
    minG = nG if require_all_groups else (min_groups if min_groups is not None else max(1, int(round(0.75*nG))))

    agg = (
        longdf_meta.groupby("feature")
                   .agg(n_groups=("group","nunique"),
                        mean_corr=("corr","mean"),
                        mean_abs=("abs_corr","mean"),
                        min_abs=("abs_corr","min"),
                        max_abs=("abs_corr","max"),
                        min_corr=("corr","min"),
                        max_corr=("corr","max"))
                   .reset_index()
    )
    kept = agg[agg["n_groups"] >= minG].copy()
    if kept.empty:
        kept = agg.copy()

    shared_all = kept.sort_values(["min_abs","mean_abs"], ascending=[False, False]).head(top_k).copy()
    kept["range_corr"] = kept["max_corr"] - kept["min_corr"]
    dissim_all = kept.sort_values("range_corr", ascending=False).head(top_k).copy()

    for c in ["mean_corr","mean_abs","min_abs","max_abs","min_corr","max_corr","range_corr"]:
        if c in shared_all.columns: shared_all[c] = shared_all[c].round(3)
        if c in dissim_all.columns: dissim_all[c] = dissim_all[c].round(3)

    shared_cols_all = ["feature","n_groups","min_abs","mean_abs","max_abs","min_corr","max_corr"]
    dissim_cols_all = ["feature","n_groups","range_corr","min_corr","max_corr"]

    with open(out_tex, "a") as f:
        f.write("%% -------- CROSS-GROUP (ALL): Top shared features --------\n")
        f.write(shared_all[shared_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} features shared across leadership × suggestions groups "
                     f"(ranked by minimum |r| across groups; {note_excluded})."),
            label="tab:lead_sugg_shared_top"
        ))
        f.write("\n\n")

        f.write("%% -------- CROSS-GROUP (ALL): Top dissimilar features --------\n")
        f.write(dissim_all[dissim_cols_all].to_latex(
            index=False, float_format="%.3f",
            caption=(f"Top {top_k} most dissimilar features across leadership × suggestions groups "
                     f"(ranked by correlation range; {note_excluded})."),
            label="tab:lead_sugg_dissimilar_top"
        ))
        f.write("\n\n")

    # ---------- 4) NEW: Pairwise comparisons between groups ----------
    if not do_pairwise or len(groups) < 2:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN GROUPS --------\n\n")

    # Pre-split per group with strength filter for pairwise logic
    per_group_df = {}
    features_to_plot = {}
    for G in groups:
        sub = longdf[longdf["group"] == G].copy()
        sub = sub[sub["abs_corr"] >= abs_thresh_pair]
        per_group_df[G] = sub.set_index("feature")[["corr","abs_corr","p","n"]]

    for A, B in itertools.combinations(groups, 2):
        A_df = per_group_df[A]
        B_df = per_group_df[B]
        if A_df.empty or B_df.empty:
            continue

        inter = A_df.join(B_df, how="inner", lsuffix="_A", rsuffix="_B")

        # keep if strong in either (as your original code does)
        inter = inter[(inter["abs_corr_A"] >= strength_thresh) | (inter["abs_corr_B"] >= strength_thresh)]
        if inter.empty:
            continue

        # signs and deltas
        inter["sign_A"] = np.where(inter["corr_A"] >= 0, 1, -1)
        inter["sign_B"] = np.where(inter["corr_B"] >= 0, 1, -1)
        inter["same_sign"] = (inter["sign_A"] == inter["sign_B"])
        inter["delta_r"] = (inter["corr_B"] - inter["corr_A"]).abs()
        inter["min_abs"] = inter[["abs_corr_A","abs_corr_B"]].min(axis=1)

        z_list, p_list = [], []
        for r in inter.itertuples():
            z, pval = fisher_pdiff_session(r.corr_A, r.n_A, r.corr_B, r.n_B)
            z_list.append(z)
            p_list.append(pval)
        inter["z_diff"] = z_list
        inter["p_diff"] = p_list

        # BH-FDR across THIS pair's comparisons
        inter["q_diff"] = _bh_fdr(inter["p_diff"])

        # Shared (same sign)
        shared = inter[inter["same_sign"]].sort_values(
            ["delta_r", "min_abs"], ascending=[True, False]
        ).head(top_k_pair).copy()

        # Dissimilar: opposite sign OR big |Δr|
        dissim_mask = (inter["delta_r"] >= diff_thresh_pair)
        dissim = inter[dissim_mask].sort_values(
            ["delta_r","min_abs"], ascending=[False, False]
        ).head(top_k_pair).copy()

        ignore = []
        for ff in dissim.index.values.tolist():

            if ff not in ignore:
                if ff not in features_to_plot.keys():
                    features_to_plot[ff] = 0
                features_to_plot[ff] += 1



        # Build output tables (now include r/p/n for each group + z/p/q)
        shared_tbl = (
            shared.reset_index()
                  [ ["feature",
                     "corr_A","p_A","n_A",
                     "corr_B","p_B","n_B",
                     "min_abs","delta_r","z_diff","p_diff","q_diff"] ]
                  .rename(columns={
                      "corr_A": f"r({A})", "p_A": f"p({A})", "n_A": f"n({A})",
                      "corr_B": f"r({B})", "p_B": f"p({B})", "n_B": f"n({B})",
                      "min_abs":"min|r|", "delta_r":"|Δr|"
                  })
        )

        dissim_tbl = (
            dissim.reset_index()
                  [ ["feature",
                     "corr_A","p_A","n_A",
                     "corr_B","p_B","n_B",
                     "min_abs","delta_r","same_sign","z_diff","p_diff","q_diff"] ]
                  .rename(columns={
                      "corr_A": f"r({A})", "p_A": f"p({A})", "n_A": f"n({A})",
                      "corr_B": f"r({B})", "p_B": f"p({B})", "n_B": f"n({B})",
                      "min_abs":"min|r|", "delta_r":"|Δr|"
                  })
        )

        # round numeric columns
        for df_ in (shared_tbl, dissim_tbl):
            for c in df_.select_dtypes(include="number").columns:
                df_[c] = df_[c].round(3)

        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {A} vs {B} — Shared (same sign, strong in both) --------\n")
            f.write(shared_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Groups {A} vs {B}: shared strong correlations "
                         f"(|r|≥{abs_thresh_pair}, same sign; ranked by min|r|; {note_excluded})."),
                label=f"tab:lead_sugg_pair_shared_{A}_vs_{B}".replace(" ","_").replace("/","-")
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {A} vs {B} — Dissimilar (opposite sign or large |Δr|) --------\n")
            f.write(dissim_tbl.to_latex(
                index=False, float_format="%.3f",
                caption=(f"Groups {A} vs {B}: dissimilar strong correlations "
                         f"(|r|≥{abs_thresh_pair} in both; opposite sign or |Δr|≥{diff_thresh_pair}; {note_excluded})."),
                label=f"tab:lead_sugg_pair_dissimilar_{A}_vs_{B}".replace(" ","_").replace("/","-")
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (including pairwise group comparisons) to: {out_tex}")
    #plot_correlations_over_sessions(longdf, features=features_to_plot,out_png="figure_sugg_trajectories.png", by="group")
    
    # 1) compute pooled (suggestion-agnostic) correlations per leader from raw data
    leader_results = compute_corrs_by_leader_with_n(raw_df,
                                                    target="performance",
                                                    leader_col="leader_type",
                                                    features=sorted(longdf["feature"].unique()))

    # 2) convert to long rows and append
    leader_all_long = leader_results_to_long(leader_results,
                                             leader_map={"ai":"AI", "robot":"AI", "human":"H"})
    longdf_aug = pd.concat([longdf, leader_all_long], ignore_index=True)
    

    raw_df2 = pd.read_csv('team_stats_ai.csv')
    AI_ONLY_LABEL = "AI*"
    #pdb.set_trace()
    # 1) Build the AI-only pooled column from raw_df2
    ai_only_long = build_single_group_long(
        raw_df2,
        label=AI_ONLY_LABEL,
        target="performance",
        features=sorted(longdf["feature"].unique().tolist()),  # align feature universe
        min_n=2,          # relax if AI-only has few rows
        fill_missing=True # keep blank cells in heatmap
    )
    
    mapping = {'human_1': 'H¹', 'human_0': 'H°', 'ai_1': 'AI¹', 'ai_0': 'AI°'}
    
    longdf["group"] = longdf["group"].replace(mapping)
    longdf_aug["group"] = longdf_aug["group"].replace(mapping)
    longdf_for_plot = pd.concat([longdf_aug, ai_only_long], ignore_index=True)
    group_order = (sorted(longdf["group"].unique().tolist())
               + [g for g in ["AI", "H", AI_ONLY_LABEL]
                  if g in longdf_for_plot["group"].unique()])


    survey_questions = ['Dependable', 'Reliable', 'Unresponsive', 'Predictable', 'Act consistently', 'Function successfully', 'Malfunction', 'Have errors', 'Provide feedback', 'Meet the needs of the mission/task', 'Provide appropriate information', 'Communicate with people', 'Perform exactly as instructed', 'Follow directions']
    demographic_questions = ['Age', 'How often do you play first-person video games on average?', 'How many times have you interacted with the TeamCollab simulation environment in the past?', 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?']
    
    prefixes = ["leader_","nonleader_"]
    features_to_remove = [*[prefix+ptt for ptt in ptt_questions for prefix in ["leader_","nonleader_", ""]], *[prefix+nasa for nasa in nasa_questions for prefix in ["leader_","nonleader_", ""]], "PTT", "nonleader_NASA", 'nonleader_How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?', "How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?", "nonleader_Dependable", *[prefix + q for prefix in prefixes for q in ["Unresponsive",'Have errors']],'Have errors', 'Perform exactly as instructed', "frac_dangerous_ambiguous", "frac_benign_as_dangerous","uncertainty_at_grabbing",'Follow directions', "leader_Predictable", "nonleader_Act consistently", "nonleader_Perform exactly as instructed", "probability_at_grabbing", "benign_ratio", 'Provide appropriate information', *[prefix + q for prefix in ["leader_","nonleader_", ""] for q in ['How often do you play first-person video games on average?']], "total_orders"]
    
    for ftr in features_to_remove:
        if ftr in features_to_plot.keys():
            del features_to_plot[ftr]
    
    '''
    # Replace feature names using the dictionary
    longdf_for_plot = longdf_for_plot[longdf_for_plot["feature"].isin(features_to_plot)].copy()
    #longdf_for_plot["feature"] = longdf_for_plot["feature"].replace(translation)

    # Function to translate with prefix handling
    def translate_with_prefix(name, translation):
        prefix = ""
        if name.startswith("leader_"):
            prefix = "leader_"
            base = name[len("leader_"):]
        elif name.startswith("nonleader_"):
            prefix = "nonleader_"
            base = name[len("nonleader_"):]
        else:
            base = name

        # Translate base if in dictionary
        translated_base = translation.get(base, base)
        return prefix + translated_base
    
    longdf_for_plot["feature"] = longdf_for_plot["feature"].apply(lambda x: translate_with_prefix(x, translation))

    # Save to CSV
    longdf_for_plot.to_csv("translated_features_sugg.csv", index=False)
    '''
    #longdf_for_plot[longdf_for_plot["feature"].isin(features_to_plot)]
    
    features_to_plot["latency_norm_median"] = 1
    features_to_plot["path_eff_median"] = 1
    pdb.set_trace()
    fig, ax = plot_corr_heatmap(
        longdf_for_plot,
        features=features_to_plot,
        annotate=True,
        by="group",
        session_order=group_order
    )
    
    
    plt.show()
    
    
def build_leader_all_long(df: pd.DataFrame,
                          target: str = "performance",
                          leader_col: str = "leader_type",
                          leader_map: dict = {"ai": "AI (all)", "human": "Human (all)"},
                          features = None,
                          min_n: int = 3) -> pd.DataFrame:
    """
    Compute pooled (ALL suggestions) Spearman correlations per leader type from raw rows.
    Returns long df with columns: ['group','feature','corr','p','n'].
    """
    num_cols = df.select_dtypes(include="number").columns.tolist()
    features = [c for c in (features or num_cols) if c != target]

    rows = []
    lead_vals = df[leader_col].astype(str).str.lower()
    for needle, out_label in leader_map.items():
        sub = df[lead_vals.str.contains(needle, na=False)].copy()
        if sub.empty:
            continue
        for feat in features:
            if feat not in sub.columns: 
                continue
            xy = sub[[feat, target]].dropna()
            n = len(xy)
            if n >= min_n and xy[feat].nunique() > 1 and xy[target].nunique() > 1:
                r, p = spearmanr(xy[feat], xy[target])
                rows.append({"group": out_label, "feature": feat, "corr": float(r), "p": float(p), "n": int(n)})
    return pd.DataFrame(rows)

def correlations_by_leadership_and_suggestions(
    df_sessions: pd.DataFrame,
    abs_threshold: Optional[float] = None,
    min_rows: int = 2,
) -> Dict[Hashable, pd.DataFrame]:
    """
    For each (leader_type, suggestions) group, compute Spearman correlations between
    `performance` and every numeric feature, returning r, p (two-sided), and n (pairwise N).

    Returns:
        Dict[(leader_type, suggestions) -> DataFrame(columns=['feature','r','p','n'])]
    """
    feat_df = df_sessions.copy()

    # --- Input checks / normalization ---
    if 'suggestions' not in feat_df.columns:
        raise ValueError("df_sessions must include 'suggestions'.")
    if 'leader_type' not in feat_df.columns:
        raise ValueError("df_sessions must include 'leader_type'.")
    if 'performance' not in feat_df.columns:
        raise ValueError("df_sessions must include 'performance'.")

    # Normalize suggestions to binary {0,1}
    feat_df['suggestions'] = feat_df['suggestions'].apply(_to_binary_suggestions)

    # Ensure performance numeric
    feat_df['performance'] = pd.to_numeric(feat_df['performance'], errors='coerce')

    # Numeric features only (exclude id-like/meta columns)
    id_like = {'team_id','session','leader_type','layout','suggestions'}
    numeric_feats = [
        c for c in feat_df.columns
        if c not in id_like and pd.api.types.is_numeric_dtype(feat_df[c])
    ]

    results: Dict[Hashable, pd.DataFrame] = {}

    # --- Group by leader_type × suggestions ---
    for (lt, sug), sub in feat_df.groupby(['leader_type','suggestions'], dropna=False):
        # Group-level sanity checks
        if len(sub) < min_rows:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','r','p','n'])
            continue
        if sub['performance'].nunique(dropna=True) <= 1:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','r','p','n'])
            continue

        # Keep features with >1 unique value in THIS group (before pairing)
        cols_here = [c for c in numeric_feats if sub[c].nunique(dropna=True) > 1]
        if not cols_here:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','r','p','n'])
            continue

        # Compute Spearman r, p, n per feature (pairwise complete cases)
        rows = []
        y = sub['performance']
        for feat in cols_here:
            x = sub[feat]

            # Pairwise complete-case mask
            m = x.notna() & y.notna()
            n = int(m.sum())
            if n < max(min_rows, 2):  # need at least 2 points for a correlation
                continue

            xv = x[m]
            yv = y[m]

            # If constant after pairing, skip
            if xv.nunique() <= 1 or yv.nunique() <= 1:
                continue

            # Spearman correlation & two-sided p-value
            r, p = spearmanr(xv, yv, nan_policy='omit')  # scipy returns (rho, pval)
            r, p = spearman_permutation_test(xv, yv, n_resamples=50000, seed=42)
            # Spearman can return nan if something degenerate happens; guard it
            if pd.isna(r):
                continue

            rows.append({"feature": feat, "r": float(r), "p": float(p), "n": n})

        if not rows:
            results[(lt, sug)] = pd.DataFrame(columns=['feature','r','p','n'])
            continue

        out = pd.DataFrame(rows)

        # Optional filter on absolute correlation
        if abs_threshold is not None:
            out = out[out['r'].abs() >= abs_threshold]

        # Sort by |r| descending (then by p ascending as a tie-breaker)
        if not out.empty:
            out = out.sort_values(by=['r'], key=lambda s: s.abs(), ascending=False)
            out = out.sort_values(by=['r','p'], key=lambda s: (s.abs() if s.name=='r' else s), ascending=[False, True])

        results[(lt, sug)] = out.reset_index(drop=True)

    return results


    
def calculation_performance_human_vs_ai(DF):

    g = DF.copy()
    #g = g[g["team_id"] != 15]
    #g = g.drop(index=62)
    g = g.replace({"human":0, "ai":1}).groupby(["team_id","leader_type"])["performance"].mean().unstack()
    order = (g.get(1, pd.Series(index=g.index, dtype=float)) - g.get(0, pd.Series(index=g.index, dtype=float))).sort_values()

    
    
    #pdb.set_trace()
    
    #Block Testing wihin mean performance------
    # Keep teams that have both leader types
    g2 = g.copy()
    if 0 not in g2.columns: g2[0] = np.nan
    if 1 not in g2.columns: g2[1] = np.nan
    g2 = g2[[0, 1]].dropna()  # rows with both Human(0) and AI(1)

    if len(g2) < 2 or g2[0].isna().all() or g2[1].isna().all():
        print("Not enough teams with both Human and AI means to run the asymmetry test.")
    else:
        df_gap = (
            g2.rename(columns={0: "human", 1: "ai"})
              .assign(gap=lambda d: d["ai"] - d["human"],
                      abs_gap=lambda d: (d["ai"] - d["human"]).abs(),
                      ai_wins=lambda d: (d["ai"] > d["human"]).astype(int))
        )

        # Group summaries
        summary = df_gap.groupby("ai_wins")["abs_gap"].agg(["mean", "std", "count"])
        print("\nAbs gap by group (0=Human wins, 1=AI wins):")
        print(summary)

        # Split samples
        pos = df_gap.loc[df_gap["ai_wins"] == 1, "abs_gap"]  # AI > Human
        neg = df_gap.loc[df_gap["ai_wins"] == 0, "abs_gap"]  # Human > AI

        if len(pos) == 0 or len(neg) == 0:
            print("\nOne group is empty—cannot run two-sample tests.")
        else:
            # Welch t-test: H1 = abs_gap(AI wins) > abs_gap(Human wins)
            tval, pval = stats.ttest_ind(pos, neg, equal_var=False, alternative="greater")

            # Mann–Whitney U (robust to non-normality), same alternative
            uval, p_u = stats.mannwhitneyu(pos, neg, alternative="greater")

            # Cohen's d (Hedges' g correction)
            def cohens_d(a, b):
                na, nb = len(a), len(b)
                sa, sb = np.var(a, ddof=1), np.var(b, ddof=1)
                s_p = np.sqrt(((na - 1) * sa + (nb - 1) * sb) / (na + nb - 2))
                if s_p == 0:
                    return 0.0
                d = (np.mean(a) - np.mean(b)) / s_p
                J = 1 - 3 / (4 * (na + nb) - 9)  # small-sample correction
                return d * J

            d = cohens_d(pos.values, neg.values)

            # Cliff's delta (nonparametric effect size)
            def cliffs_delta(x, y):
                x = np.asarray(x); y = np.asarray(y)
                gt = sum((xi > y).sum() for xi in x)
                lt = sum((xi < y).sum() for xi in x)
                return (gt - lt) / (len(x) * len(y))

            delta = cliffs_delta(pos.values, neg.values)

            print(f"\nWelch t-test (abs_gap | AI_wins > Human_wins): t = {tval:.3f}, p = {pval:.4g}")
            print(f"Mann–Whitney U (greater): U = {uval:.1f}, p = {p_u:.4g}")
            print(f"Effect sizes: Cohen's d = {d:.2f}, Cliff's delta = {delta:.2f}")

            # Optional: quick textual takeaway
            mean_ai = summary.loc[1, "mean"] if 1 in summary.index else np.nan
            mean_h  = summary.loc[0, "mean"] if 0 in summary.index else np.nan
            print(f"\nMean |gap| when AI wins: {mean_ai:.2f}  vs  when Human wins: {mean_h:.2f}")


    pos = df_gap.loc[df_gap["ai_wins"]==1, "abs_gap"].values  # AI wins
    neg = df_gap.loc[df_gap["ai_wins"]==0, "abs_gap"].values  # Human wins
    obs = pos.mean() - neg.mean()

    combined = np.concatenate([pos, neg])
    n_pos = len(pos)

    iters = 100_000
    count = 0
    for _ in range(iters):
        np.random.shuffle(combined)
        new_pos = combined[:n_pos]
        new_neg = combined[n_pos:]
        if new_pos.mean() - new_neg.mean() >= obs:
            count += 1

    p_perm = (count + 1) / (iters + 1)  # one-sided
    print("Permutation p (one-sided):", p_perm)

    rng = np.random.default_rng(0)
    B = 100_000
    boot = []
    for _ in range(B):
        boot.append(rng.choice(pos, size=len(pos), replace=True).mean()
                   - rng.choice(neg, size=len(neg), replace=True).mean())
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    print("Bootstrap 95% CI for mean(|gap|_AIwins - |gap|_Humanwins):", (ci_lo, ci_hi))


def fmt_layouts(lst):
    # e.g., [1,3,4] -> "L1,L3,L4"; handle empty/missing
    if not lst:
        return "L–"
    return "L" + ",L".join(str(x) for x in lst)

def fmt_names(lst):
    # e.g., [1,3,4] -> "L1,L3,L4"; handle empty/missing
    if not lst:
        return "–"
    return "[" + ", ".join(str(x) for x in lst) + "]"
def plot_mean_differences_leader_type(DF):

    # Build: per (team_id, leader_type) -> list of layouts where they led
    layouts_by_team_leader = (
        DF.groupby(["team_id", "leader_type"])["layout"]
          .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
          .to_dict()
    )
    names_by_team_leader = (
        DF.groupby(["team_id", "leader_type"])["name"]
          .apply(lambda s: sorted(pd.unique(s.dropna().astype(int))))
          .to_dict()
    )

    g = DF.copy().replace({"human":0, "ai":1}).groupby(["team_id","leader_type"])["dangerous_weighted_work"].mean().unstack()
    order = (g.get(1, pd.Series(index=g.index, dtype=float)) - g.get(0, pd.Series(index=g.index, dtype=float))).sort_values()

    ypos = range(len(order))

    # scatter points
    plt.scatter(g.loc[order.index, 0], ypos, marker='o', label="Human")
    plt.scatter(g.loc[order.index, 1], ypos, marker='s', label="AI")

    # connecting lines + annotations
    for i, t in enumerate(order.index):
        x_h = g.loc[t, 0] if 0 in g.columns and t in g.index else np.nan
        x_a = g.loc[t, 1] if 1 in g.columns and t in g.index else np.nan

        # connect if both exist
        if pd.notna(x_h) and pd.notna(x_a):
            plt.plot([x_h, x_a], [i, i], color='0.6', linewidth=1)

        # annotate layouts next to Human point (left side)
        if pd.notna(x_h):
            labs_h = fmt_layouts(layouts_by_team_leader.get((t, 0), []))
            nabs_h = fmt_names(names_by_team_leader.get((t, 0), []))
            #plt.annotate(nabs_h, (x_h, i), xytext=(-6, -2), textcoords='offset points',
            #             ha='right', va='center', fontsize=8, color='C0')

        # annotate layouts next to AI point (right side)
        if pd.notna(x_a):
            labs_a = fmt_layouts(layouts_by_team_leader.get((t, 1), []))
            nabs_a = fmt_names(names_by_team_leader.get((t, 1), []))
            #plt.annotate(nabs_a, (x_a, i), xytext=(6, -2), textcoords='offset points',
            #             ha='left', va='center', fontsize=8, color='C1')

    plt.yticks(ypos, order.index)
    plt.xlabel("Mean performance")
    plt.ylabel("Team #")
    plt.title("Within-team performance: Human vs AI")
    plt.legend()
    plt.tight_layout()
    plt.show()


def analyze_ai_advantage_print(
    fe_df: pd.DataFrame,
    grain: str = "team_layout",          # "team_layout" (prefer) or "team"
    permutations: int = 20000,
    bootstrap_reps: int = 10000,
    seed: int = 42,
    save_tables: bool = False,
    out_dir: str = "ai_vs_human_outputs",
):
    """
    Prints main results for: 'Does AI help more when human-led performance is worse?'

    Inputs
    ------
    fe_df : DataFrame with columns:
        - 'team_id' (required), optional 'layout'
        - 'leader_type' in {'human','ai'} (case-insensitive)
        - 'performance' (your metric; not unitless if you've chosen so)
    grain : "team_layout" tries to pair within (team_id, layout). Falls back to "team"
            when no pairs exist. Use "team" to force team-level pairs.
    permutations : # of permutations for slope permutation test
    bootstrap_reps : # of bootstrap reps for quartile CIs of mean Δ
    seed : RNG seed
    save_tables : if True, writes CSVs for pairs/summary/quartiles into out_dir
    out_dir : directory used when save_tables=True

    Returns
    -------
    results : dict with 'pairs', 'summary' (DataFrame), 'quartiles' (DataFrame),
              'ols_model', 'logit_model', and 'perm' (dict with slopes & p-values).
    """
    rng = np.random.default_rng(seed)
    df = fe_df.copy()
    df["leader_type"] = df["leader_type"].astype(str).str.strip().str.lower()
    df = df[df["leader_type"].isin(["human", "ai"])]

    def build_pairs(df, grain):
        # Prefer team×layout pairs if available
        if grain == "team_layout" and "layout" in df.columns:
            p = (
                df.pivot_table(index=["team_id","layout"],
                               columns="leader_type",
                               values="performance",
                               aggfunc="mean")
                  .dropna(subset=["human","ai"])
                  .reset_index()
                  .rename(columns={"human":"perf_human","ai":"perf_ai"})
            )
            if len(p) > 0:
                p["pair_grain"] = "team_layout"
                return p
        # fallback to team-level
        p = (
            df.pivot_table(index=["team_id"],
                           columns="leader_type",
                           values="performance",
                           aggfunc="mean")
              .dropna(subset=["human","ai"])
              .reset_index()
              .rename(columns={"human":"perf_human","ai":"perf_ai"})
        )
        p["pair_grain"] = "team"
        return p

    pairs = build_pairs(df, grain)
    if pairs.empty:
        raise ValueError("No paired human/ai observations found at the chosen grain.")

    pairs["delta"] = pairs["perf_ai"] - pairs["perf_human"]
    pairs["ai_win"] = (pairs["delta"] > 0).astype(int)
    grain_used = pairs["pair_grain"].iloc[0]
    n_pairs = len(pairs)

    # ---------- OLS: Δ ~ human_baseline ----------
    ols = smf.ols("delta ~ perf_human", data=pairs).fit()
    slope = float(ols.params.get("perf_human", np.nan))
    slope_p_two_sided = float(ols.pvalues.get("perf_human", np.nan))
    ci = ols.conf_int().loc["perf_human"].to_numpy()
    r2 = float(ols.rsquared)
    # One-sided p-value for H1: slope < 0 (based on OLS t-stat sign)
    if slope < 0:
        p_one_sided_ols = slope_p_two_sided / 2
    else:
        p_one_sided_ols = 1 - slope_p_two_sided / 2

    # ---------- Spearman ----------
    spearman_r, spearman_p = stats.spearmanr(pairs["perf_human"], pairs["delta"])

    # ---------- Logistic: P(AI win) ~ human_baseline ----------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            logit = smf.logit("ai_win ~ perf_human", data=pairs).fit(disp=False)
            logit_coef = float(logit.params.get("perf_human", np.nan))
            logit_p = float(logit.pvalues.get("perf_human", np.nan))
            # Odds ratio and a simple probability contrast (Q1 vs Q4)
            xq = pairs["perf_human"].quantile([0.25, 0.75]).to_numpy()
            xb_q1 = float(logit.params["Intercept"] + logit_coef * xq[0])
            xb_q4 = float(logit.params["Intercept"] + logit_coef * xq[1])
            p_q1 = 1/(1+np.exp(-xb_q1))
            p_q4 = 1/(1+np.exp(-xb_q4))
            odds_ratio = float(np.exp(logit_coef))
        except Exception:
            logit = None
            logit_coef = np.nan
            logit_p = np.nan
            odds_ratio = np.nan
            p_q1 = np.nan
            p_q4 = np.nan

    # ---------- Permutation test for slope (one-sided: slope < 0) ----------
    x = pairs["perf_human"].to_numpy()
    y = pairs["delta"].to_numpy()

    def slope_with_intercept(x, y):
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return float(beta[1])

    obs_slope = slope_with_intercept(x, y)
    slopes = np.empty(permutations)
    for i in range(permutations):
        y_perm = rng.permutation(y)
        slopes[i] = slope_with_intercept(x, y_perm)
    p_perm_one_sided = (slopes <= obs_slope).mean() if obs_slope < 0 else (slopes >= obs_slope).mean()
    p_perm_two_sided = (np.abs(slopes) >= np.abs(obs_slope)).mean()

    # ---------- Quartiles of human baseline & bootstrap CIs for mean Δ ----------
    q_bins = min(4, pairs["perf_human"].nunique())
    pairs["human_q"] = pd.qcut(pairs["perf_human"], q=q_bins, duplicates="drop")
    rows = []
    for qlabel, grp in pairs.groupby("human_q"):
        deltas = grp["delta"].to_numpy()
        wins = grp["ai_win"].to_numpy()
        n = len(deltas)
        mean_delta = float(np.mean(deltas)) if n else np.nan
        mean_human = float(np.mean(grp["perf_human"])) if n else np.nan
        ai_win_rate = float(np.mean(wins)) if n else np.nan
        if n >= 2:
            idx = rng.integers(0, n, size=(bootstrap_reps, n))
            boot_means = deltas[idx].mean(axis=1)
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
        else:
            ci_lo = ci_hi = np.nan
        rows.append({
            "quartile": str(qlabel),
            "n": n,
            "mean_human": mean_human,
            "mean_delta": mean_delta,
            "delta_CI_lo": float(ci_lo),
            "delta_CI_hi": float(ci_hi),
            "ai_win_rate": ai_win_rate
        })
    quart_tbl = pd.DataFrame(rows).sort_values("mean_human")

    # ---------- Printing helpers ----------
    def fmt(x, nd=3):
        return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

    def star(p):
        if not np.isfinite(p): return ""
        return " ***" if p < 0.001 else \
               " **"  if p < 0.01  else \
               " *"   if p < 0.05  else \
               " †"   if p < 0.10  else ""

    # ---------- PRINT MAIN RESULTS ----------
    print("\n=== AI advantage vs Human baseline: Main Results ===")
    print(f"Pair grain used       : {grain_used}")
    print(f"Number of pairs (n)   : {n_pairs}")

    print("\n[OLS] Δ (AI−Human) ~ Human baseline")
    print(f"  Slope (β)           : {fmt(slope)} "
          f"(95% CI {fmt(ci[0])}, {fmt(ci[1])}), R² = {fmt(r2)}")
    print(f"  p (two-sided)       : {fmt(slope_p_two_sided)}{star(slope_p_two_sided)}")
    print(f"  p (one-sided, β<0)  : {fmt(p_one_sided_ols)}{star(p_one_sided_ols)}")

    print("\n[Permutation test for slope] H1: β < 0")
    print(f"  Observed slope      : {fmt(obs_slope)}")
    print(f"  p_perm (one-sided)  : {fmt(p_perm_one_sided)}{star(p_perm_one_sided)}")
    print(f"  p_perm (two-sided)  : {fmt(p_perm_two_sided)}{star(p_perm_two_sided)}")

    print("\n[Spearman correlation]")
    print(f"  ρ(human, Δ)         : {fmt(spearman_r)}")
    print(f"  p-value             : {fmt(spearman_p)}{star(spearman_p)}")

    print("\n[Logistic] P(AI win) ~ Human baseline")
    if np.isfinite(logit_coef):
        print(f"  Coef                : {fmt(logit_coef)} (OR = {fmt(np.exp(logit_coef))})")
        print(f"  p-value             : {fmt(logit_p)}{star(logit_p)}")
        print(f"  Pred P(AI win) @ Q1 : {fmt(p_q1)}; @ Q4 : {fmt(p_q4)}")
    else:
        print("  (logistic model did not converge or insufficient variation)")

    print("\n[Quartiles of human baseline]  Mean Δ (AI−Human) with 95% bootstrap CI")
    for _, r in quart_tbl.iterrows():
        print(f"  {r['quartile']}: n={int(r['n'])}, "
              f"mean_human={fmt(r['mean_human'])}, "
              f"mean_Δ={fmt(r['mean_delta'])} "
              f"[{fmt(r['delta_CI_lo'])}, {fmt(r['delta_CI_hi'])}], "
              f"AI-win rate={fmt(r['ai_win_rate'])}")

    # ---------- Optional: save tables ----------
    if save_tables:
        os.makedirs(out_dir, exist_ok=True)
        pairs.to_csv(os.path.join(out_dir, "pairs.csv"), index=False)
        quart_tbl.to_csv(os.path.join(out_dir, "quartiles.csv"), index=False)
        # concise summary table
        summary = pd.DataFrame({
            "metric": [
                "pairs_n","grain_used","OLS_slope","OLS_p(two-sided)","OLS_R2",
                "perm_p(one-sided)","perm_p(two-sided)","Spearman_r","Spearman_p",
                "Logit_coef","Logit_p","Logit_OR","Pred_P_Q1","Pred_P_Q4"
            ],
            "value": [
                n_pairs, grain_used, slope, slope_p_two_sided, r2,
                p_perm_one_sided, p_perm_two_sided, spearman_r, spearman_p,
                logit_coef, logit_p, odds_ratio, p_q1, p_q4
            ]
        })
        summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
        print(f"\nSaved CSVs to: {out_dir}/ (pairs.csv, quartiles.csv, summary.csv)")

    # ---------- Return objects ----------
    summary_df = pd.DataFrame({
        "metric": [
            "pairs_n","grain_used","OLS_slope","OLS_p(two-sided)","OLS_R2",
            "perm_p(one-sided)","perm_p(two-sided)","Spearman_r","Spearman_p",
            "Logit_coef","Logit_p"
        ],
        "value": [
            n_pairs, grain_used, slope, slope_p_two_sided, r2,
            p_perm_one_sided, p_perm_two_sided, spearman_r, spearman_p,
            logit_coef, logit_p
        ]
    })
    return {
        "pairs": pairs,
        "summary": summary_df,
        "quartiles": quart_tbl,
        "ols_model": ols,
        "logit_model": logit,
        "perm": {
            "slopes": slopes,
            "obs_slope": obs_slope,
            "p_one_sided": p_perm_one_sided,
            "p_two_sided": p_perm_two_sided,
        },
    }


def compute_corrs_by_leader(df, target="performance", leader_col="leader_type", top_n=10, output_print=True):


    # Numeric columns except the target
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    factors = [col for col in numeric_cols if col != target]

    def compute_corrs(group):
        results = []
        for col in factors:
            valid = group[[col, target]].dropna()
            if valid.shape[0] > 1:
                rho, pval = spearmanr(valid[col], valid[target])
                results.append((col, rho, pval))
        df_corr = pd.DataFrame(results, columns=["factor", "rho", "pval"])
        # Sort: strongest |rho| first, then lowest p-value
        df_corr = df_corr.sort_values(
            by=["rho", "pval"],
            key=lambda x: x.abs() if x.name=="rho" else x,
            ascending=[False, True]   # largest |rho| first, then smallest p-value
        )
        return df_corr

    results = {}
    for leader, group in df.groupby(leader_col):
        results[leader] = compute_corrs(group)

    if output_print:
        # Print top N for each leader type
        for leader, corr_df in results.items():
            print(f"\n=== Top {top_n} correlations for leader_type = {leader} ===")
            print(corr_df.head(top_n).to_string(index=False))

    return results

def compute_corrs_by_leader_with_n(
    df: pd.DataFrame,
    target: str = "performance",
    leader_col: str = "leader_type",
    features: Optional[Iterable[Hashable]] = None,
    min_n: int = 3,
) -> Dict[str, pd.DataFrame]:
    """
    Per-leader pooled Spearman correlations of each feature with `target`.
    Returns: {leader_value_lowercased: DataFrame[feature, corr, p, n]}
    """

    # --- helpers ---
    def _is_single_col_name(c) -> bool:
        # Accept hashable scalars (e.g., str, int, tuple for MultiIndex), reject list-likes
        return pd.api.types.is_hashable(c) and not isinstance(c, (list, tuple, np.ndarray))

    def _varies(x) -> bool:
        # robust scalar check of "more than one unique value"
        if isinstance(x, pd.DataFrame):
            # if someone accidentally passed multiple cols, require ANY column to vary
            return bool((x.nunique() > 1).any())
        # Series/array-like → scalar int
        try:
            return pd.Series(x).nunique() > 1
        except Exception:
            return False

    # --- feature list (numeric by default, excluding target) ---
    if features is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        features = [c for c in numeric_cols if c != target]
    else:
        # sanitize externally-provided features
        features = [c for c in features if _is_single_col_name(c) and c in df.columns and c != target]

    results: Dict[str, pd.DataFrame] = {}
    # Keep NaNs as their own group if present
    for leader, group in df.groupby(leader_col, dropna=False):
        rows: List[dict] = []
        for feat in features:
            # pull just the two columns and drop NA
            xy = group[[feat, target]].dropna()
            n = int(len(xy))
            if n < max(min_n, 3):
                continue

            x = xy[feat]          # should be a Series (single column)
            y = xy[target]

            if not (_varies(x) and _varies(y)):
                continue

            rho, pval = spearmanr(x, y)
            rho, pval = spearman_permutation_test(x, y, n_resamples=50000, seed=42)
            rows.append({"feature": feat, "corr": float(rho), "p": float(pval), "n": n})

        results[str(leader).lower() if pd.notna(leader) else "nan"] = pd.DataFrame(rows)

    return results

def build_single_group_long(df: pd.DataFrame,
                            label: str,
                            target: str = "performance",
                            features=None,
                            min_n: int = 3,
                            fill_missing: bool = True) -> pd.DataFrame:
    """
    Compute Spearman r(feature, target) over the entire df for a single 'group' label.
    Always emits one row per requested feature. If correlation cannot be computed,
    fills corr/p with NaN and reports n (number of valid pairs) if fill_missing=True.

    Returns columns: ['group','feature','corr','p','n'].
    """
    cols = ["group","feature","corr","p","n"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)

    # Choose features (numeric except target) unless provided
    if features is None:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        features = [c for c in num_cols if c != target]
    else:
        # Dedupe while preserving order; keep only existing columns
        seen = set()
        features = [c for c in features
                    if c != target and c in df.columns and not (c in seen or seen.add(c))]

    out_rows = []

    for feat in features:
        # Pull columns & coerce numeric (so strings won't break)
        sub = df[[feat, target]].copy()
        sub[feat] = pd.to_numeric(sub[feat], errors="coerce")
        sub[target] = pd.to_numeric(sub[target], errors="coerce")
        sub = sub.dropna()

        # Handle duplicate column labels: df[feat] could be a DataFrame.
        x_obj = df[feat]
        if isinstance(x_obj, pd.DataFrame):
            # Collapse duplicates row-wise (mean across duplicate columns)
            x_series = x_obj.apply(pd.to_numeric, errors="coerce").mean(axis=1)
            sub = pd.concat([x_series, df[target]], axis=1).rename(columns={0: feat})
            sub = sub.apply(pd.to_numeric, errors="coerce").dropna()

        n = int(len(sub))
        can_corr = (n >= min_n and
                    sub[feat].nunique() > 1 and
                    sub[target].nunique() > 1)

        if can_corr:
            r, p = spearmanr(sub[feat], sub[target])
            r, p = spearman_permutation_test(sub[feat], sub[target], n_resamples=50000, seed=42)
            out_rows.append({"group": label, "feature": feat, "corr": float(r), "p": float(p), "n": n})
        else:
            if fill_missing:
                # Emit a placeholder row so the heatmap keeps this cell as blank
                out_rows.append({"group": label, "feature": feat, "corr": np.nan, "p": np.nan, "n": n})
            # If not fill_missing, we simply skip this feature.

    return pd.DataFrame(out_rows, columns=cols)
    
    
def leader_results_to_long(leader_results: Dict[str, pd.DataFrame],
                           leader_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Convert {'ai': df, 'human': df, ...} to long format expected by heatmap:
    columns: ['group','feature','corr','p','n']
    """
    leader_map = leader_map or {"ai": "AI (all)", "robot": "AI (all)", "human": "Human (all)"}
    out = []
    for key, dfk in leader_results.items():
        if key in leader_map and dfk is not None and not dfk.empty:
            tmp = dfk.copy()
            tmp["group"] = leader_map[key]
            out.append(tmp[["group","feature","corr","p","n"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["group","feature","corr","p","n"])

def leader_results_to_long(leader_results, leader_map=None):
    """
    Convert {'ai': df, 'human': df, ...} -> long format with group labels.
    leader_map maps raw keys to heatmap column names.
    """
    leader_map = leader_map or {"ai": "AI (all)", "robot": "AI (all)", "human": "Human (all)"}
    out = []
    for key, dfk in leader_results.items():
        if key in leader_map and dfk is not None and not dfk.empty:
            tmp = dfk.copy()
            tmp["group"] = leader_map[key]
            out.append(tmp[["group","feature","corr","p","n"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["group","feature","corr","p","n"])


def partial_spearman(x, y, controls_df=None):
    """
    Approximate partial Spearman by:
      1) rank-transform x and y
      2) regress rank(x) ~ controls, rank(y) ~ controls
      3) Pearson corr between residuals
    Falls back to plain Spearman if no controls provided.
    """
    if controls_df is None or controls_df.shape[1] == 0:
        rho, p = spearmanr(x, y, nan_policy="omit")
        return rho, p

    # rank-transform
    xr = pd.Series(x).rank(method="average")
    yr = pd.Series(y).rank(method="average")

    # Build design matrix for controls (add intercept)
    # Use patsy to handle categorical covariates nicely.
    # Example: controls formula "0 + C(layout) + session" if you pass columns as they are.
    X = controls_df.copy()
    # Create a formula like: "0 + col1 + col2 + C(catcol)" when col is object or categorical
    # We’ll auto-detect categorical/object columns and wrap with C()
    terms = []
    for c in X.columns:
        if pd.api.types.is_categorical_dtype(X[c]) or X[c].dtype == "object":
            terms.append(f"C({c})")
        else:
            terms.append(c)
    formula = "0 + " + " + ".join(terms) if terms else "0 + 1"
    X_design = patsy.dmatrix(formula, X, return_type="dataframe")

    # Drop rows with any NaNs
    valid = (~xr.isna()) & (~yr.isna()) & (~X_design.isna().any(axis=1))
    xr = xr[valid]
    yr = yr[valid]
    X_design = X_design.loc[valid]

    if xr.size < 3:  # not enough data
        return np.nan, np.nan

    # OLS residuals
    # resid_x = xr - X @ (X^+ xr)
    # Use numpy lstsq (least squares)
    beta_x, *_ = np.linalg.lstsq(X_design.values, xr.values, rcond=None)
    beta_y, *_ = np.linalg.lstsq(X_design.values, yr.values, rcond=None)
    resid_x = xr.values - X_design.values @ beta_x
    resid_y = yr.values - X_design.values @ beta_y

    # Pearson on residuals
    if resid_x.std(ddof=1) == 0 or resid_y.std(ddof=1) == 0:
        return np.nan, np.nan
    rho = np.corrcoef(resid_x, resid_y)[0,1]

    # For p-value, approximate with t distribution for Pearson:
    n = len(resid_x)
    if n < 4:
        return rho, np.nan
    t = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-12))
    # two-sided p
    from scipy.stats import t as tdist
    p = 2 * (1 - tdist.cdf(np.abs(t), df=n-2))
    return rho, p

def correlations_by_leader2(
    df,
    target="performance",
    leader_col="leader_type",
    exclude_contains=("relative_performance",),  # add substrings to exclude leakage/duplicates
    control_cols=None,  # e.g., ["layout"] or ["layout", "session"]
    top_n=30
):

    # Normalize leader labels
    df[leader_col] = df[leader_col].str.lower().str.strip()

    # Build exclusion mask for columns
    def include_col(c):
        if c == target:
            return False
        if any(sub in c for sub in exclude_contains):
            return False
        return True

    # Numeric factors only (you can add object dummies via control_cols)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    factors = [c for c in numeric_cols if include_col(c)]

    # Prepare controls (optional)
    controls = None
    if control_cols:
        # Keep as DataFrame with those columns
        controls = df[control_cols].copy()

    def compute_for_group(group):
        # Identify which columns are usable (not constant, have at least 3 non-NaNs paired with target)
        results = []
        for col in factors:
            valid = group[[col, target]].dropna()
            if valid.shape[0] < 3:
                continue
            # drop constant columns in this group
            if valid[col].nunique(dropna=True) < 2 or valid[target].nunique(dropna=True) < 2:
                continue

            # controls aligned to valid rows if provided
            ctrl = None
            if controls is not None:
                idx = valid.index
                ctrl = controls.loc[idx]
            rho, p = partial_spearman(valid[col], valid[target], controls_df=ctrl)
            n = len(valid)
            results.append((col, n, rho, p))

        if not results:
            return pd.DataFrame(columns=["factor","n","rho","pval","qval"])

        out = pd.DataFrame(results, columns=["factor","n","rho","pval"])

        # Multiple testing correction (BH/FDR)
        mask = out["pval"].notna()
        qvals = np.full(out.shape[0], np.nan)
        if mask.any():
            _, q, _, _ = multipletests(out.loc[mask, "pval"].values, method="fdr_bh")
            qvals[mask.values] = q
        out["qval"] = qvals

        # Sort: strongest |rho|, then smallest q, then smallest p
        out = out.sort_values(
            by=["rho", "qval", "pval"],
            key=lambda x: x.abs() if x.name == "rho" else x,
            ascending=[False, True, True]
        ).reset_index(drop=True)
        return out

    # Compute per leader
    for leader, g in df.groupby(leader_col):
        res = compute_for_group(g)
        print(f"\n=== Top {top_n} correlations for leader_type = {leader} ===")
        if res.empty:
            print("(no valid factors)")
            continue
        # Pretty print
        display = res.head(top_n).copy()
        # Add simple significance stars based on q (FDR)
        def stars(q):
            if pd.isna(q): return ""
            if q < 0.001: return "***"
            if q < 0.01:  return "**"
            if q < 0.05:  return "*"
            return ""
        display["sig"] = display["qval"].apply(stars)
        cols = ["factor","n","rho","pval","qval","sig"]
        print(display[cols].to_string(index=False))


def partial_spearman2(x, y, controls_df=None):
    """
    Approximate partial Spearman:
      1) rank-transform x and y
      2) regress rank(x) ~ controls, rank(y) ~ controls (OLS)
      3) Pearson corr between residuals; p via t-approximation
    Returns (rho, pval, n_used)
    """
    s = pd.concat([pd.Series(x), pd.Series(y)], axis=1).dropna()
    if s.shape[0] < 3: 
        return np.nan, np.nan, s.shape[0]
    xr = s.iloc[:,0].rank(method="average")
    yr = s.iloc[:,1].rank(method="average")
    n = len(xr)

    if controls_df is None or controls_df.shape[1] == 0:
        rho, p = spearmanr(xr, yr)
        return rho, p, n

    # Align controls and add intercept; one-hot for categoricals
    Xc = controls_df.loc[s.index].copy()
    X = pd.get_dummies(Xc, drop_first=True)
    X["intercept"] = 1.0

    # OLS residuals
    bx, *_ = np.linalg.lstsq(X.values, xr.values, rcond=None)
    by, *_ = np.linalg.lstsq(X.values, yr.values, rcond=None)
    rx = xr.values - X.values @ bx
    ry = yr.values - X.values @ by

    if rx.std(ddof=1) == 0 or ry.std(ddof=1) == 0:
        return np.nan, np.nan, n

    rho = np.corrcoef(rx, ry)[0, 1]
    # two-sided p via Pearson t-test approx
    if n < 4 or abs(rho) >= 1:
        return rho, np.nan, n
    t = rho * np.sqrt((n - 2) / (1 - rho**2 + 1e-12))
    p = 2 * (1 - tdist.cdf(abs(t), df=n-2))
    return rho, p, n

def fisher_z(r):
    # bound to avoid ±inf for |r|=1 from rounding
    r = max(min(r, 0.999999), -0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))

def _normalize_simple_boolish(x):
    """Map many truthy/falsey representations to 'with'/'without'.
       Leaves NaN as NaN so it becomes its own group if present."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    truthy = {"1","true","t","yes","y","on","with","enabled","suggestions","suggestion","suggested"}
    falsy  = {"0","false","f","no","n","off","without","disabled","none","nosuggestions","no_suggestions"}
    if s in truthy:  return "with"
    if s in falsy:   return "without"
    # fallback: keep original token so you can see unexpected labels
    return s

def correlations_by_leader_and_suggestions(
    df,
    target="performance",
    leader_col="leader_type",
    suggestions_col="suggestions",
    exclude_contains=("relative_performance",),   # exclude leakage/duplicates
    control_cols=None,                             # e.g., ["layout"] or ["layout","session"]
    top_n=30
):
    # --- Normalize group labels ---
    df = df.copy()
    df[leader_col] = df[leader_col].astype(str).str.lower().str.strip()
    df[suggestions_col] = df[suggestions_col].apply(_normalize_simple_boolish)

    # --- Column inclusion filter ---
    def include_col(c):
        if c == target:
            return False
        if any(sub in c for sub in exclude_contains or []):
            return False
        return True

    # Numeric candidate factors
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    factors = [c for c in numeric_cols if include_col(c)]

    # Optional controls
    controls = None
    if control_cols:
        controls = df[control_cols].copy()

    def compute_for_group(group):
        rows = []
        for col in factors:
            valid = group[[col, target]].dropna()
            if valid.shape[0] < 3:
                continue
            # drop columns that are constant within this group (or constant target)
            if valid[col].nunique(dropna=True) < 2 or valid[target].nunique(dropna=True) < 2:
                continue

            # Align controls to valid rows if provided
            ctrl = None
            if controls is not None:
                ctrl = controls.loc[valid.index]

            rho, p = partial_spearman(valid[col], valid[target], controls_df=ctrl)
            rows.append((col, len(valid), rho, p))

        if not rows:
            return pd.DataFrame(columns=["factor","n","rho","pval","qval"])

        out = pd.DataFrame(rows, columns=["factor","n","rho","pval"])

        # FDR (BH) within the group
        mask = out["pval"].notna()
        qvals = np.full(out.shape[0], np.nan)
        if mask.any():
            _, q, _, _ = multipletests(out.loc[mask, "pval"].values, method="fdr_bh")
            qvals[mask.values] = q
        out["qval"] = qvals

        # Sort by |rho| desc, then q asc, then p asc
        out = out.sort_values(
            by=["rho","qval","pval"],
            key=lambda x: x.abs() if x.name == "rho" else x,
            ascending=[False, True, True]
        ).reset_index(drop=True)
        return out

    # Pretty stars
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001:  return "***"
        if q < 0.01:   return "**"
        if q < 0.05:   return "*"
        return ""

    # --- Compute per (leader, suggestions) group ---
    results = {}  # (leader, suggestions) -> DataFrame
    # dropna=False so NaN suggestions (if any) form their own bucket and are not silently dropped
    for (leader, sugg), g in df.groupby([leader_col, suggestions_col], dropna=False):
        res = compute_for_group(g)
        results[(leader, sugg)] = res

        print(f"\n=== Top {top_n} correlations for leader_type = {leader!r} | suggestions = {sugg!r} ===")
        if res.empty:
            print("(no valid factors)")
            continue
        display = res.head(top_n).copy()
        display["sig"] = display["qval"].apply(stars)
        print(display[["factor","n","rho","pval","qval","sig"]].to_string(index=False))

    return results

def compare_ai_vs_human_deltas(
    df,
    target="performance",
    leader_col="leader_type",
    team_col="team_id",
    order_col="session",               # used to order the 2 sessions (2 per leader per team)
    exclude_contains=("relative_performance",),
    control_mode=None,                 # None | "layout" | "complexity" | ["layout","session"], etc.
    top_n=30
):
    """
    For each team and each leader type ('ai'/'human'), this computes per-team deltas:
        Δfactor = factor(second session) - factor(first session)
        Δtarget = target(second session) - target(first session)
    Then correlates Δfactor vs Δtarget separately for AI-led and Human-led groups,
    and compares correlations via Fisher's z test.

    Assumptions:
      - Each team has exactly 2 AI-led and 2 Human-led rows (if more, the earliest and latest by `order_col` are used).
      - `order_col` is sortable (e.g., 1..4 or timestamps).
      - `control_mode`, if provided, will also be delta-coded (numeric controls only).
    """
    df = df.copy()
    df[leader_col] = df[leader_col].str.lower().str.strip()

    # Numeric factors (excluding target and any leaky/duplicate features)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    def keep(col):
        if col == target: return False
        if any(sub in col for sub in exclude_contains): return False
        return True
    factors = [c for c in num_cols if keep(c)]

    # Determine which controls to use
    if control_mode is None:
        control_cols = []
    else:
        control_cols = [control_mode] if isinstance(control_mode, str) else list(control_mode)

    # We'll delta-code ONLY numeric controls (categorical deltas are ill-defined)
    numeric_controls = [c for c in control_cols if c in num_cols and c != target]

    # Helper: for a (team, leader) group, return a single-row Series of deltas
    def _two_point_delta(g: pd.DataFrame):
        g_sorted = g.sort_values(order_col)
        first, last = g_sorted.iloc[0], g_sorted.iloc[-1]
        out = {}

        # Target delta
        if pd.notna(first[target]) and pd.notna(last[target]):
            out[f"Δ_{target}"] = last[target] - first[target]
        else:
            out[f"Δ_{target}"] = np.nan

        # Factor deltas
        for c in factors:
            v1, v2 = first[c], last[c]
            out[f"Δ_{c}"] = (v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan

        # Control deltas
        for c in numeric_controls:
            v1, v2 = first[c], last[c]
            out[f"Δ_ctrl__{c}"] = (v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan

        return pd.Series(out)

    # Build per-team deltas for AI and Human separately
    # Keep only teams that have at least 2 rows within each leader type
    def _build_deltas_for_leader(df_lead: pd.DataFrame) -> pd.DataFrame:
        # filter to teams with >=2 rows
        counts = df_lead.groupby(team_col).size()
        valid_teams = counts[counts >= 2].index
        dd = (df_lead[df_lead[team_col].isin(valid_teams)]
              .groupby(team_col, as_index=True)
              .apply(_two_point_delta))
        # Drop teams where Δtarget is missing
        if isinstance(dd, pd.DataFrame):
            dd = dd.dropna(subset=[f"Δ_{target}"], how="any")
        return dd

    g_ai = df[df[leader_col] == "ai"]
    g_hu = df[df[leader_col] == "human"]

    if g_ai.empty or g_hu.empty:
        raise ValueError("Both 'ai' and 'human' groups are required in leader_type.")

    d_ai = _build_deltas_for_leader(g_ai)
    d_hu = _build_deltas_for_leader(g_hu)

    # If too few teams, bail early
    if d_ai is None or d_hu is None or d_ai.shape[0] < 4 or d_hu.shape[0] < 4:
        return pd.DataFrame()

    # For each factor, compute partial Spearman on deltas
    results = []
    for col in factors:
        x_ai = d_ai[f"Δ_{col}"]
        y_ai = d_ai[f"Δ_{target}"]
        x_hu = d_hu[f"Δ_{col}"]
        y_hu = d_hu[f"Δ_{target}"]

        # Build Δ-controls, align indices
        if numeric_controls:
            c_ai = d_ai[[f"Δ_ctrl__{c}" for c in numeric_controls]]
            c_hu = d_hu[[f"Δ_ctrl__{c}" for c in numeric_controls]]
        else:
            c_ai = None
            c_hu = None

        # Drop NaNs per group
        ai_valid = pd.concat([x_ai, y_ai], axis=1).dropna()
        hu_valid = pd.concat([x_hu, y_hu], axis=1).dropna()

        if ai_valid.shape[0] < 4 or hu_valid.shape[0] < 4:
            continue
        if ai_valid.iloc[:,0].nunique() < 2 or ai_valid.iloc[:,1].nunique() < 2:
            continue
        if hu_valid.iloc[:,0].nunique() < 2 or hu_valid.iloc[:,1].nunique() < 2:
            continue

        # Align control rows to valid indices
        a_ctrl = c_ai.loc[ai_valid.index] if c_ai is not None else None
        h_ctrl = c_hu.loc[hu_valid.index] if c_hu is not None else None

        rho_ai, p_ai, n_ai = partial_spearman2(ai_valid.iloc[:,0], ai_valid.iloc[:,1], a_ctrl)
        rho_hu, p_hu, n_hu = partial_spearman2(hu_valid.iloc[:,0], hu_valid.iloc[:,1], h_ctrl)

        if np.isnan(rho_ai) or np.isnan(rho_hu) or n_ai < 4 or n_hu < 4:
            continue

        # Fisher z-test for difference between independent correlations
        z_ai = fisher_z(rho_ai)
        z_hu = fisher_z(rho_hu)
        se = np.sqrt(1/(n_ai - 3) + 1/(n_hu - 3))
        z_stat = (z_ai - z_hu) / se
        p_diff = 2 * (1 - norm.cdf(abs(z_stat)))

        results.append((
            col, n_ai, rho_ai, p_ai, n_hu, rho_hu, p_hu,
            rho_ai - rho_hu, z_stat, p_diff
        ))

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results, columns=[
        "factor", "n_ai", "rho_ai", "p_ai", "n_hu", "rho_hu", "p_hu",
        "delta_rho", "z_diff", "p_diff"
    ])

    # FDR over p_diff
    mask = out["p_diff"].notna()
    qvals = np.full(out.shape[0], np.nan)
    if mask.any():
        _, q, _, _ = multipletests(out.loc[mask, "p_diff"], method="fdr_bh")
        qvals[mask.values] = q
    out["q_diff"] = qvals

    # sort by |Δρ| desc, then q asc, then p asc
    out = out.sort_values(
        by=["delta_rho", "q_diff", "p_diff"],
        key=lambda x: x.abs() if x.name == "delta_rho" else x,
        ascending=[False, True, True]
    ).reset_index(drop=True)

    # Pretty preview
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return ""
    view = out.head(top_n).copy()
    view["sig"] = view["q_diff"].apply(stars)

    print(f"\n=== Δ( factor ) vs Δ( {target} ): AI vs Human (controls on Δ of {numeric_controls if numeric_controls else 'None'}) ===")
    cols_to_show = ["factor","n_ai","rho_ai","n_hu","rho_hu","delta_rho","p_diff","q_diff","sig"]
    print(view[cols_to_show].to_string(index=False))
    return out

# ---------- helpers ----------


def _rank_series(x: pd.Series) -> np.ndarray:
    # Spearman residualization works on ranks
    return pd.Series(x).rank(method="average").to_numpy(dtype=float)

def _partial_spearman(x: pd.Series, y: pd.Series, controls):
    """
    Partial Spearman via rank-residualization.
    If controls is None or empty -> plain Spearman.
    Returns (rho, pval, n)
    """
    s = pd.concat([x, y, controls] if controls is not None else [x, y], axis=1).dropna()
    if s.shape[0] < 4:
        return (np.nan, np.nan, s.shape[0])

    xr = _rank_series(s.iloc[:, 0])
    yr = _rank_series(s.iloc[:, 1])

    if controls is None or controls.shape[1] == 0:
        rho, p = spearmanr(xr, yr)
        return (rho, p, len(xr))

    Cr = np.column_stack([_rank_series(s.iloc[:, 2 + j]) for j in range(controls.shape[1])])

    # residualize xr ~ Cr ; yr ~ Cr
    reg = LinearRegression()
    reg.fit(Cr, xr)
    xr_res = xr - reg.predict(Cr)
    reg.fit(Cr, yr)
    yr_res = yr - reg.predict(Cr)

    rho, p = spearmanr(xr_res, yr_res)
    return (rho, p, len(xr))

# ---------- main ----------
def compare_two_groups_human_deltas(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    target="performance",
    leader_col="leader_type",
    team_col="team_id",
    order_col="session",               # used to order the 2+ human-led sessions per team
    exclude_contains=("relative_performance",),
    control_mode=None,                 # None | "layout" | "complexity" | ["layout","session"], etc.
    min_n=4,                           # minimum valid teams per group to keep a factor
    top_n=30
) -> pd.DataFrame:
    """
    For each dataframe (A and B), restrict to HUMAN-led rows, compute per-team deltas:
        Δfactor = factor(last human-led session) - factor(first human-led session)
        Δtarget = target(last) - target(first)
    Then compute (partial) Spearman correlations of Δfactor vs Δtarget separately for A and B,
    and compare correlations using Fisher's z for independent samples.

    Assumptions:
      - Each team has >= 2 human-led rows (if more, first and last by order_col are used).
      - order_col is sortable (ints or timestamps).
      - Controls (if any) are numeric and delta-coded as well.
    """
    def _prep(df):
        d = df.copy()
        d[leader_col] = d[leader_col].astype(str).str.lower().str.strip()
        d = d[d[leader_col] == "human"]

        num_cols = d.select_dtypes(include="number").columns.tolist()

        def keep(col):
            if col == target: return False
            if any(sub in col for sub in exclude_contains): return False
            return True

        factors = [c for c in num_cols if keep(c)]

        if control_mode is None:
            ctrl_cols = []
        else:
            ctrl_cols = [control_mode] if isinstance(control_mode, str) else list(control_mode)
        numeric_controls = [c for c in ctrl_cols if c in num_cols and c != target]

        def _two_point_delta(g: pd.DataFrame):
            g_sorted = g.sort_values(order_col)
            first, last = g_sorted.iloc[0], g_sorted.iloc[-1]
            out = {f"Δ_{target}": (last[target] - first[target]) if pd.notna(first[target]) and pd.notna(last[target]) else np.nan}
            for c in factors:
                v1, v2 = first[c], last[c]
                out[f"Δ_{c}"] = (v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan
            for c in numeric_controls:
                v1, v2 = first[c], last[c]
                out[f"Δ_ctrl__{c}"] = (v2 - v1) if pd.notna(v1) and pd.notna(v2) else np.nan
            return pd.Series(out)

        # keep only teams with >=2 human-led rows
        counts = d.groupby(team_col).size()
        valid = counts[counts >= 2].index
        if len(valid) == 0:
            return pd.DataFrame(), factors, numeric_controls

        dd = (d[d[team_col].isin(valid)]
              .groupby(team_col, as_index=True)
              .apply(_two_point_delta))

        if isinstance(dd, pd.DataFrame):
            dd = dd.dropna(subset=[f"Δ_{target}"], how="any")

        return dd, factors, numeric_controls

    dA, factorsA, ctrlsA = _prep(df_a)
    dB, factorsB, ctrlsB = _prep(df_b)

    # Only compare on the intersection of factors present in BOTH groups
    common_factors = sorted(set(factorsA).intersection(factorsB))

    # Controls: use intersection as well (safe, symmetric)
    common_ctrls = sorted(set(ctrlsA).intersection(ctrlsB))
    A_ctrl_cols = [f"Δ_ctrl__{c}" for c in common_ctrls] if common_ctrls else []
    B_ctrl_cols = [f"Δ_ctrl__{c}" for c in common_ctrls] if common_ctrls else []

    results = []

    for col in common_factors:
        xa = dA.get(f"Δ_{col}", pd.Series(dtype=float))
        ya = dA.get(f"Δ_{target}", pd.Series(dtype=float))
        xb = dB.get(f"Δ_{col}", pd.Series(dtype=float))
        yb = dB.get(f"Δ_{target}", pd.Series(dtype=float))

        # Build controls if any (align by index after dropping NaNs inside _partial_spearman)
        Ca = dA[A_ctrl_cols] if A_ctrl_cols else None
        Cb = dB[B_ctrl_cols] if B_ctrl_cols else None

        rho_a, p_a, n_a = _partial_spearman(xa, ya, Ca)
        rho_b, p_b, n_b = _partial_spearman(xb, yb, Cb)

        # ensure usable sample sizes and non-degenerate variance
        if (n_a is None or n_b is None or n_a < min_n or n_b < min_n
            or np.isnan(rho_a) or np.isnan(rho_b)):
            continue

        # Fisher z for independent correlations
        za = fisher_z(rho_a)
        zb = fisher_z(rho_b)
        se = np.sqrt(1/(n_a - 3) + 1/(n_b - 3))
        z_stat = (za - zb) / se
        p_diff = 2 * (1 - norm.cdf(abs(z_stat)))

        results.append((
            col, n_a, rho_a, p_a, n_b, rho_b, p_b,
            rho_a - rho_b, z_stat, p_diff
        ))

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results, columns=[
        "factor", "n_A", "rho_A", "p_A", "n_B", "rho_B", "p_B",
        "delta_rho(A−B)", "z_diff", "p_diff"
    ])

    # Multiple testing correction over the cross-group difference
    mask = out["p_diff"].notna()
    qvals = np.full(out.shape[0], np.nan)
    if mask.any():
        _, q, _, _ = multipletests(out.loc[mask, "p_diff"], method="fdr_bh")
        qvals[mask.values] = q
    out["q_diff"] = qvals

    # Nice sorting: biggest |Δρ| first, then by q, then p
    out = out.sort_values(
        by=["delta_rho(A−B)", "q_diff", "p_diff"],
        key=lambda s: s.abs() if s.name == "delta_rho(A−B)" else s,
        ascending=[False, True, True]
    ).reset_index(drop=True)

    # Optional pretty preview
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return ""

    view = out.head(top_n).copy()
    view["sig"] = view["q_diff"].apply(stars)
    print(f"\n=== Δ(factor) vs Δ({target}) on HUMAN-led sessions: DataFrame A vs DataFrame B "
          f"(Δ-controls: {common_ctrls if common_ctrls else 'None'}) ===")
    print(view[["factor","n_A","rho_A","n_B","rho_B","delta_rho(A−B)","p_diff","q_diff","sig"]].to_string(index=False))

    return out

# ------------------------------------------------------------
# Main comparison
# ------------------------------------------------------------
def compare_ai_vs_human(
    df,
    target="performance",
    leader_col="leader_type",
    exclude_contains=("relative_performance",),  # exclude leak/duplicate features
    control_mode=None,   # None | "layout" | "complexity" | ["layout","session"], etc.
    top_n=30
):
    """
    control_mode:
      - None: plain Spearman
      - str or list[str]: columns to control for (e.g., "layout", "complexity", or ["layout","session"])
    """
    df[leader_col] = df[leader_col].str.lower().str.strip()

    # numeric factors only (excluding the target & leakage)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    def keep(col):
        if col == target:
            return False
        if any(sub in col for sub in exclude_contains):
            return False
        return True
    factors = [c for c in num_cols if keep(c)]

    # controls
    if control_mode is None:
        controls = None
    else:
        if isinstance(control_mode, str):
            control_cols = [control_mode]
        else:
            control_cols = list(control_mode)
        controls = df[control_cols].copy()

    # split groups
    groups = dict(tuple(df.groupby(leader_col)))
    if "ai" not in groups or "human" not in groups:
        raise ValueError("Both 'ai' and 'human' groups are required in leader_type.")

    g_ai = groups["ai"].copy()
    g_hu = groups["human"].copy()

    results = []
    for col in factors:
        # Build aligned controls per factor after NaN drop
        a_valid = g_ai[[col, target]].dropna()
        h_valid = g_hu[[col, target]].dropna()

        if a_valid.shape[0] < 4 or h_valid.shape[0] < 4:
            continue
        if a_valid[col].nunique() < 2 or a_valid[target].nunique() < 2:
            continue
        if h_valid[col].nunique() < 2 or h_valid[target].nunique() < 2:
            continue

        a_ctrl = controls.loc[a_valid.index] if controls is not None else None
        h_ctrl = controls.loc[h_valid.index] if controls is not None else None

        rho_ai, p_ai, n_ai = partial_spearman2(a_valid[col], a_valid[target], a_ctrl)
        rho_hu, p_hu, n_hu = partial_spearman2(h_valid[col], h_valid[target], h_ctrl)

        if np.isnan(rho_ai) or np.isnan(rho_hu) or n_ai < 4 or n_hu < 4:
            continue

        # Fisher z-test for difference in independent correlations
        z_ai = fisher_z(rho_ai)
        z_hu = fisher_z(rho_hu)
        se = np.sqrt(1/(n_ai - 3) + 1/(n_hu - 3))
        z_stat = (z_ai - z_hu) / se
        p_diff = 2 * (1 - norm.cdf(abs(z_stat)))

        results.append((
            col, n_ai, rho_ai, p_ai, n_hu, rho_hu, p_hu,
            rho_ai - rho_hu, z_stat, p_diff
        ))

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results, columns=[
        "factor", "n_ai", "rho_ai", "p_ai", "n_hu", "rho_hu", "p_hu",
        "delta_rho", "z_diff", "p_diff"
    ])

    # FDR for the difference p-values
    mask = out["p_diff"].notna()
    qvals = np.full(out.shape[0], np.nan)
    if mask.any():
        _, q, _, _ = multipletests(out.loc[mask,"p_diff"], method="fdr_bh")
        qvals[mask.values] = q
    out["q_diff"] = qvals

    # sort by |Δρ| desc, then q_diff asc, then p_diff asc
    out = out.sort_values(
        by=["delta_rho", "q_diff", "p_diff"],
        key=lambda x: x.abs() if x.name == "delta_rho" else x,
        ascending=[False, True, True]
    ).reset_index(drop=True)

    # pretty print top_n
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return ""
    view = out.head(top_n).copy()
    view["sig"] = view["q_diff"].apply(stars)

    print(f"\n=== AI vs Human: correlation differences (control={control_mode}) ===")
    cols_to_show = ["factor","n_ai","rho_ai","n_hu","rho_hu","delta_rho","p_diff","q_diff","sig"]
    print(view[cols_to_show].to_string(index=False))
    return out

def _normalize_suggestions_token(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    truthy = {"1","true","t","yes","y","on","with","enabled","suggestions","suggestion","suggested"}
    falsy  = {"0","false","f","no","n","off","without","disabled","none","nosuggestions","no_suggestions"}
    if s in truthy: return "with"
    if s in falsy:  return "without"
    return s  # keep unexpected label as-is

def compare_suggestions_within_leader(
    df: pd.DataFrame,
    target: str = "performance",
    leader_col: str = "leader_type",
    suggestions_col: str = "suggestions",
    exclude_contains=("relative_performance",),   # exclude leak/duplicate features
    control_mode=None,                             # None | "layout" | "complexity" | ["layout","session"], etc.
    min_n: int = 4,
    top_n: int = 30,
):
    """
    Compares correlations with the target between suggestions='with' vs 'without' within each leader group.
    Produces two tables: one for leader='ai' and one for leader='human'.

    control_mode:
      - None: plain Spearman (via partial_spearman2 with controls_df=None)
      - str or list[str]: columns to control for (e.g., "layout" or ["layout","session"])
    """
    df = df.copy()
    # normalize labels
    df[leader_col] = df[leader_col].astype(str).str.lower().str.strip()
    if suggestions_col not in df.columns:
        raise ValueError(f"Missing {suggestions_col!r} column")
    df[suggestions_col] = df[suggestions_col].apply(_normalize_suggestions_token)

    # factor selection: numeric only, no target, no leakage
    num_cols = df.select_dtypes(include="number").columns.tolist()
    def keep(col):
        if col == target:
            return False
        if any(sub in col for sub in (exclude_contains or ())):
            return False
        return True
    factors = [c for c in num_cols if keep(c)]
    if not factors:
        raise ValueError("No numeric factors found after exclusions.")

    # controls
    if control_mode is None:
        controls_df = None
    else:
        control_cols = [control_mode] if isinstance(control_mode, str) else list(control_mode)
        missing = [c for c in control_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing control columns: {missing}")
        controls_df = df[control_cols].copy()

    # helper: compute with/without suggestions within a given leader subgroup
    def compute_for_leader(leader_label: str):
        sub_leader = df[df[leader_col] == leader_label]
        if sub_leader.empty:
            return pd.DataFrame()

        groups = dict(tuple(sub_leader.groupby(suggestions_col, dropna=False)))
        if "with" not in groups or "without" not in groups:
            return pd.DataFrame()  # need both sides

        g_with = groups["with"].copy()
        g_without = groups["without"].copy()

        rows = []
        for col in factors:
            w_valid = g_with[[col, target]].dropna()
            u_valid = g_without[[col, target]].dropna()

            # min size & non-constant checks within *aligned* valid pairs
            if w_valid.shape[0] < min_n or u_valid.shape[0] < min_n:
                continue
            if w_valid[col].nunique(dropna=True) < 2 or w_valid[target].nunique(dropna=True) < 2:
                continue
            if u_valid[col].nunique(dropna=True) < 2 or u_valid[target].nunique(dropna=True) < 2:
                continue

            w_ctrl = controls_df.loc[w_valid.index] if controls_df is not None else None
            u_ctrl = controls_df.loc[u_valid.index] if controls_df is not None else None

            rho_w, p_w, n_w = partial_spearman2(w_valid[col], w_valid[target], w_ctrl)
            rho_u, p_u, n_u = partial_spearman2(u_valid[col], u_valid[target], u_ctrl)

            if any(np.isnan([rho_w, rho_u])) or n_w < min_n or n_u < min_n:
                continue

            # Fisher z test for difference
            z_w = fisher_z(rho_w)
            z_u = fisher_z(rho_u)
            se = np.sqrt(1/(n_w - 3) + 1/(n_u - 3))
            z_stat = (z_w - z_u) / se
            p_diff = 2 * (1 - norm.cdf(abs(z_stat)))

            rows.append((col, n_w, rho_w, p_w, n_u, rho_u, p_u, rho_w - rho_u, z_stat, p_diff))

        if not rows:
            return pd.DataFrame()

        out = pd.DataFrame(rows, columns=[
            "factor", "n_with", "rho_with", "p_with", "n_without", "rho_without", "p_without",
            "delta_rho_with_minus_without", "z_diff", "p_diff"
        ])

        # FDR on difference p-values within this leader
        mask = out["p_diff"].notna()
        qvals = np.full(out.shape[0], np.nan)
        if mask.any():
            _, q, _, _ = multipletests(out.loc[mask, "p_diff"].values, method="fdr_bh")
            qvals[mask.values] = q
        out["q_diff"] = qvals
        out["leader"] = leader_label

        # sort by |Δρ| then q then p
        out = out.sort_values(
            by=["delta_rho_with_minus_without", "q_diff", "p_diff"],
            key=lambda x: x.abs() if x.name == "delta_rho_with_minus_without" else x,
            ascending=[False, True, True]
        ).reset_index(drop=True)
        return out

    res_ai = compute_for_leader("ai")
    res_hu = compute_for_leader("human")

    def _print_block(df_block, leader_label):
        print(f"\n=== Suggestions effect within leader = {leader_label} (control={control_mode}) ===")
        if df_block is None or df_block.empty:
            print("(no valid factors)")
            return
        def stars(q):
            if pd.isna(q): return ""
            if q < 0.001: return "***"
            if q < 0.01:  return "**"
            if q < 0.05:  return "*"
            return ""
        view = df_block.head(top_n).copy()
        view["sig"] = view["q_diff"].apply(stars)
        cols = ["factor","n_with","rho_with","n_without","rho_without",
                "delta_rho_with_minus_without","p_diff","q_diff","sig"]
        print(view[cols].to_string(index=False))

    _print_block(res_ai, "ai")
    _print_block(res_hu, "human")

    # Return combined results (stacked) for further analysis
    combined = pd.concat([r for r in [res_ai, res_hu] if r is not None and not r.empty], ignore_index=True) \
               if (res_ai is not None or res_hu is not None) else pd.DataFrame()
    return combined


def correlations_with_conditions(
    df,
    target="performance",
    leader_col="leader_type",
    suggestions_col="suggestions",
    exclude_contains=("relative_performance",),
    top_n=30,
    min_n=4,
):
    """
    Computes Spearman correlations between all numeric features and the four
    leader_type × suggestions conditions (ai_with, ai_without, human_with, human_without).

    Returns a DataFrame of all results, prints top correlations.
    """

    df = df.copy()
    # normalize
    df[leader_col] = df[leader_col].astype(str).str.lower().str.strip()
    df[suggestions_col] = df[suggestions_col].astype(str).str.lower().str.strip()
    df["condition"] = df[leader_col] + "_" + df[suggestions_col]

    # one-hot encode condition
    cond_dummies = pd.get_dummies(df["condition"], prefix="cond")

    # numeric factors
    num_cols = df.select_dtypes(include="number").columns.tolist()
    def keep(col):
        if col == target: return False
        if any(sub in col for sub in (exclude_contains or ())): return False
        return True
    factors = [c for c in num_cols if keep(c)]

    results = []
    for col in factors:
        for dummy in cond_dummies.columns:
            valid = df[[col]].join(cond_dummies[[dummy]]).dropna()
            if valid.shape[0] < min_n: 
                continue
            if valid[col].nunique() < 2:
                continue
            rho, p = spearmanr(valid[col], valid[dummy])
            results.append((col, dummy, len(valid), rho, p))

    if not results:
        print("No valid correlations computed.")
        return pd.DataFrame()

    out = pd.DataFrame(results, columns=["feature","condition","n","rho","pval"])

    # FDR correction
    mask = out["pval"].notna()
    qvals = np.full(out.shape[0], np.nan)
    if mask.any():
        _, q, _, _ = multipletests(out.loc[mask,"pval"], method="fdr_bh")
        qvals[mask.values] = q
    out["qval"] = qvals

    # sort
    out = out.sort_values(
        by=["rho","qval","pval"],
        key=lambda x: x.abs() if x.name=="rho" else x,
        ascending=[False,True,True]
    ).reset_index(drop=True)

    # pretty print
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return ""
    view = out.head(top_n).copy()
    view["sig"] = view["qval"].apply(stars)

    print("\n=== Correlations between numeric features and condition dummies ===")
    print(view[["feature","condition","n","rho","pval","qval","sig"]].to_string(index=False))

    return out

def _norm_leader(s):
    return str(s).strip().lower()

def _norm_suggestions(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().lower()
    truthy = {"1","true","t","yes","y","on","with","enabled","suggestions","suggestion","suggested"}
    falsy  = {"0","false","f","no","n","off","without","disabled","none","nosuggestions","no_suggestions"}
    if s in truthy: return "with"
    if s in falsy:  return "without"
    return s

def _cliffs_delta(x, y):
    # Returns Cliff's delta in [-1,1]
    # Efficient computation using ranks
    x = np.asarray(x); y = np.asarray(y)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0: return np.nan
    # pairwise compare via broadcasting if small; fallback to rank trick for larger
    # Rank trick: delta = (2*AUC - 1), where AUC is Mann-Whitney U / (nx*ny)
    U = mannwhitneyu(x, y, alternative='two-sided').statistic
    auc = U / (nx * ny)
    return 2*auc - 1

def _residualize(series, controls_df):
    """OLS residualize series ~ controls (adds const). Returns residuals aligned to index."""
    y = series.astype(float)
    X = controls_df.copy()
    X = sm.add_constant(pd.get_dummies(X, drop_first=True, dtype=float))
    # drop rows with NA in either
    valid = y.index.intersection(X.dropna().index)
    if len(valid) < 3:
        # not enough to fit; return centered series as fallback
        return series - series.mean()
    yv = y.loc[valid]
    Xv = X.loc[valid]
    try:
        model = sm.OLS(yv, Xv, missing="drop").fit()
        resid = pd.Series(index=series.index, dtype=float)
        resid.loc[valid] = yv - model.predict(Xv)
        return resid
    except Exception:
        # fallback if OLS fails
        return series - series.mean()

# ------------ main ------------
def compare_suggestions_within_each_leader_by_feature(
    df: pd.DataFrame,
    target: str = "performance",
    leader_col: str = "leader_type",
    suggestions_col: str = "suggestions",
    exclude_contains=("relative_performance",),   # exclude leak/duplicate features
    control_mode=None,                             # None | str | list[str] (e.g., "session" or ["layout","session"])
    min_n: int = 4,
    top_n: int = 25,
    sort_by: str = "q",                            # "q" or "effect" (|Cliff’s δ|)
    corr_method: str = "spearman",                 # "spearman" or "pearson"
):
    """
    For each leader ('ai', 'human'), compare WITH vs WITHOUT suggestions across numeric features.

    Adds per-condition correlations (WITH and WITHOUT) of each feature with `target`,
    plus Fisher r-to-z test for the difference.

    Tests included:
      - Mann–Whitney U (two-sided) on residualized feature values between WITH vs WITHOUT
        -> columns: p (MW), q (BH/FDR per leader), Cliff's delta, mean/median diffs (original scale)
      - Correlations within each condition (WITH vs WITHOUT) with `target`
        -> columns: rho_with, p_with, rho_without, p_without
      - Fisher r-to-z difference test (WITH vs WITHOUT)
        -> columns: delta_r, z_diff, p_diff, q_diff (BH/FDR per leader)
    """
    # ---- helpers ----
    def _safe_atanh(r):
        # clamp to open interval (-1,1) for numerical stability
        eps = 1e-12
        return np.arctanh(np.clip(r, -1 + eps, 1 - eps))

    def _fisher_z_diff(r1, n1, r2, n2):
        # returns z, p (two-sided); None, None if not computable
        if min(n1, n2) <= 3:
            return np.nan, np.nan
        if np.isfinite(r1) and np.isfinite(r2):
            z1 = _safe_atanh(r1)
            z2 = _safe_atanh(r2)
            se = np.sqrt(1/(n1-3) + 1/(n2-3))
            if se == 0 or not np.isfinite(se):
                return np.nan, np.nan
            z = (z1 - z2) / se
            # two-sided p from z
            p = 2 * (1 - 0.5 * (1 + erf(abs(z)/np.sqrt(2))))  # use error function
            return float(z), float(p)
        return np.nan, np.nan

    # error function without importing scipy.special to keep deps tight
    from math import erf

    def _safe_corr(x, y, method="spearman"):
        # returns rho, p, n after dropping pairwise NaNs and constants
        xy = pd.concat([x, y], axis=1).dropna()
        n = len(xy)
        if n < min_n:
            return np.nan, np.nan, n
        if xy.iloc[:, 0].nunique() <= 1 or xy.iloc[:, 1].nunique() <= 1:
            return np.nan, np.nan, n
        if method == "pearson":
            r, p = pearsonr(xy.iloc[:, 0].values, xy.iloc[:, 1].values)
        else:
            r, p = spearmanr(xy.iloc[:, 0].values, xy.iloc[:, 1].values)
        return float(r), float(p), n

    df = df.copy()
    df[leader_col] = df[leader_col].map(_norm_leader)
    df[suggestions_col] = df[suggestions_col].map(_norm_suggestions)

    # numeric factors (excluding target & leak)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    def keep(c):
        if c == target: return False
        if any(sub in c for sub in (exclude_contains or ())): return False
        return True
    features = [c for c in num_cols if keep(c)]
    leaders = ["ai", "human"]

    # controls
    if control_mode is None:
        control_cols = None
    else:
        control_cols = [control_mode] if isinstance(control_mode, str) else list(control_mode)
        missing = [c for c in control_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing control columns: {missing}")

    all_rows = []
    for leader in leaders:
        subL = df[df[leader_col] == leader].copy()
        if subL.empty:
            continue

        # residualize features AND target if controlling
        resid_features = {}
        if control_cols:
            ctrl = subL[control_cols]
            for feat in features:
                resid_features[feat] = _residualize(subL[feat], ctrl)
            resid_target = _residualize(subL[target], ctrl)
        else:
            for feat in features:
                resid_features[feat] = subL[feat]
            resid_target = subL[target]

        # Split by suggestions
        idx_with = subL[subL[suggestions_col] == "with"].index
        idx_without = subL[subL[suggestions_col] == "without"].index

        for feat in features:
            x_with = resid_features[feat].loc[idx_with].dropna()
            x_without = resid_features[feat].loc[idx_without].dropna()

            n_with, n_without = len(x_with), len(x_without)
            if n_with < min_n or n_without < min_n:
                continue

            # skip degenerate (both constant)
            if pd.Series(x_with).nunique(dropna=True) <= 1 and pd.Series(x_without).nunique(dropna=True) <= 1:
                continue

            # Mann–Whitney U (two-sided) on residualized feature
            try:
                stat, p_mw = mannwhitneyu(x_with, x_without, alternative="two-sided")
            except ValueError:
                # identical numbers in both groups or other degenerate
                continue

            # Cliff's delta on residualized values
            d = _cliffs_delta(x_with.values, x_without.values)

            # mean/median diffs on original scale
            xo = df.loc[idx_with, feat].dropna()
            yo = df.loc[idx_without, feat].dropna()
            if len(xo) >= min_n and len(yo) >= min_n:
                mean_diff = float(xo.mean() - yo.mean())
                median_diff = float(xo.median() - yo.median())
            else:
                mean_diff = np.nan
                median_diff = np.nan

            # --- correlations within each condition (feature vs target) ---
            rho_with, p_with, n_corr_with = _safe_corr(
                resid_features[feat].loc[idx_with], resid_target.loc[idx_with], method=corr_method
            )
            rho_without, p_without, n_corr_without = _safe_corr(
                resid_features[feat].loc[idx_without], resid_target.loc[idx_without], method=corr_method
            )

            # Fisher r-to-z difference (WITH vs WITHOUT)
            if np.isfinite(rho_with) and np.isfinite(rho_without):
                z_diff, p_diff = _fisher_z_diff(rho_with, n_corr_with, rho_without, n_corr_without)
                delta_r = float(rho_with - rho_without)
            else:
                z_diff, p_diff, delta_r = np.nan, np.nan, np.nan

            all_rows.append({
                "leader": leader,
                "feature": feat,
                # sample sizes for MW
                "n_with": n_with,
                "n_without": n_without,
                # Mann–Whitney
                "p": p_mw,
                "cliffs_delta": d,
                "mean_diff_with_minus_without": mean_diff,
                "median_diff_with_minus_without": median_diff,
                # correlations per condition
                "rho_with": rho_with,
                "p_with": p_with,
                "n_corr_with": n_corr_with,
                "rho_without": rho_without,
                "p_without": p_without,
                "n_corr_without": n_corr_without,
                # correlation difference
                "delta_r": delta_r,
                "z_diff": z_diff,
                "p_diff": p_diff,
            })

    if not all_rows:
        print("No valid comparisons.")
        return pd.DataFrame()

    out = pd.DataFrame(all_rows)

    # FDR per leader for MW p-values -> 'q'
    out["q"] = np.nan
    for leader in leaders:
        mask = out["leader"].eq(leader) & out["p"].notna()
        if mask.any():
            _, qvals, _, _ = multipletests(out.loc[mask, "p"].values, method="fdr_bh")
            out.loc[mask, "q"] = qvals

    # FDR per leader for CORR DIFF p-values -> 'q_diff'
    out["q_diff"] = np.nan
    for leader in leaders:
        mask = out["leader"].eq(leader) & out["p_diff"].notna()
        if mask.any():
            _, qvals, _, _ = multipletests(out.loc[mask, "p_diff"].values, method="fdr_bh")
            out.loc[mask, "q_diff"] = qvals

    # Sorting (primary by q or by |cliffs_delta|), then by correlation diff stats for tie-breaks
    if sort_by == "effect":
        out = out.sort_values(
            by=["leader", out["cliffs_delta"].abs().name, "q", "p", out["delta_r"].abs().name, "q_diff", "p_diff"],
            ascending=[True, False, True, True, False, True, True]
        )
    else:  # "q"
        out = out.sort_values(
            by=["leader", "q", "p", out["delta_r"].abs().name, "q_diff", "p_diff", out["cliffs_delta"].abs().name],
            ascending=[True, True, True, False, True, True, False]
        )

    # Pretty print top-N per leader
    def stars(q):
        if pd.isna(q): return ""
        if q < 0.001: return "***"
        if q < 0.01:  return "**"
        if q < 0.05:  return "*"
        return ""

    for leader in leaders:
        block = out[out["leader"] == leader].head(top_n).copy()
        if block.empty:
            print(f"\n=== {leader.upper()} (with vs without) — no valid results ===")
            continue
        block["sig"] = block["q"].apply(stars)
        block["sig_diff"] = block["q_diff"].apply(stars)

        cols = [
            "feature",
            # sample sizes for MW
            "n_with","n_without",
            # per-condition correlations
            "rho_with","p_with","rho_without","p_without","delta_r","z_diff","p_diff","q_diff","sig_diff",
            # MW test & effect
            "cliffs_delta","mean_diff_with_minus_without","median_diff_with_minus_without","p","q","sig",
        ]

        print(f"\n=== {leader.upper()} leadership: WITH vs WITHOUT suggestions"
              f"{' (residualized: '+', '.join(control_cols)+')' if control_cols else ''} ===")
        print(block[cols].to_string(index=False))

    return out

def first_last_predictiveness(fe_df,
                              team_col="team_id",
                              session_col="session",
                              perf_col="performance",
                              leader_col="leader_type"):
    # Identify first/last session indices
    s_first = fe_df[session_col].min()
    s_last  = fe_df[session_col].max()

    # Build first/last tables
    first = (fe_df[fe_df[session_col] == s_first]
             [[team_col, perf_col, leader_col]].rename(columns={perf_col: "perf_first"}))
    last  = (fe_df[fe_df[session_col] == s_last]
             [[team_col, perf_col, leader_col]].rename(columns={perf_col: "perf_last"}))

    # Align teams present in both endpoints (inner join)
    endpts = pd.merge(first[[team_col, "perf_first", leader_col]],
                      last [[team_col, "perf_last" , leader_col]],
                      on=team_col, suffixes=("_first","_last"))

    # If leader_type could differ across sessions, prefer last’s label; fall back to first
    if leader_col in endpts.columns:
        endpts[leader_col] = endpts[f"{leader_col}_y"] if f"{leader_col}_y" in endpts.columns else endpts[leader_col]
        # Clean up possible duplicate columns
        for c in [f"{leader_col}_x", f"{leader_col}_y"]:
            if c in endpts.columns: endpts.drop(columns=c, inplace=True)

    def corr_and_reg(df, label):
        if len(df) < 3:
            print(f"[{label}] Not enough teams (n={len(df)}) to compute stats.")
            return
        # Correlations
        rho, p_s = spearmanr(df["perf_first"], df["perf_last"])
        r,   p_p = pearsonr (df["perf_first"], df["perf_last"])
        # OLS
        mdl = smf.ols("perf_last ~ perf_first", data=df).fit()
        print(f"\n=== {label} === (n={len(df)})")
        print(f"Spearman rho = {rho:.3f}, p = {p_s:.3f}")
        print(f"Pearson  r   = {r:.3f}, p = {p_p:.3f}")
        print(f"OLS: perf_last ~ perf_first | beta_first = {mdl.params.get('perf_first', float('nan')):.3f}, "
              f"p = {mdl.pvalues.get('perf_first', float('nan')):.3f}, R² = {mdl.rsquared:.3f}")

    # Overall
    corr_and_reg(endpts, "Overall")
    # By leadership type (if available)
    if leader_col in endpts.columns:
        for lt, sub in endpts.groupby(leader_col):
            corr_and_reg(sub.copy(), f"{leader_col}={lt}")

# =========================
# 1) Identify the subset: teams where Human mean > AI mean
# =========================
def identify_human_better(fe_df,
                          team_col="team_id",
                          leader_col="leader_type",
                          perf_col="performance"):
    # mean performance by team × leader
    g = (fe_df.groupby([team_col, leader_col])[perf_col]
         .mean().unstack(leader_col))
    # Expect columns "AI" and "Human"
    missing_cols = [c for c in ["ai","human"] if c not in g.columns]
    if missing_cols:
        raise ValueError(f"Missing expected leader types {missing_cols} in {leader_col}. Found: {list(g.columns)}")

    human_better_ids = g.index[g["human"] > g["ai"]].tolist()
    ai_better_ids    = g.index[g["ai"] > g["human"]].tolist()
    ties_ids         = g.index[g["ai"] == g["human"]].tolist()
    return g, human_better_ids, ai_better_ids, ties_ids

# =========================
# 2) Session-by-session comparison within the subset
# =========================
def session_summary(fe_df, team_ids, leader, team_col="team_id", leader_col="leader_type",
                    perf_col="performance", session_col="session"):
    df = fe_df[(fe_df[team_col].isin(team_ids)) & (fe_df[leader_col]==leader)]
    summ = (df.groupby(session_col)[perf_col]
              .agg(["mean","median","std","count"]).reset_index())
    return df, summ

def paired_step_tests(df, team_col="team_id", session_col="session", perf_col="performance"):
    sessions = sorted(df[session_col].unique())
    rows = []
    for s1, s2 in zip(sessions[:-1], sessions[1:]):
        a = df[df[session_col]==s1].sort_values(team_col)[[team_col, perf_col]]
        b = df[df[session_col]==s2].sort_values(team_col)[[team_col, perf_col]]
        common = a.merge(b, on=team_col, suffixes=("_s1","_s2"))
        if len(common)==0: 
            rows.append({"comparison": f"{s1}→{s2}", "n_pairs": 0})
            continue
        diff = common[f"{perf_col}_s2"] - common[f"{perf_col}_s1"]
        try:
            stat, p = wilcoxon(common[f"{perf_col}_s1"], common[f"{perf_col}_s2"])
        except ValueError:
            # All zero diffs case, Wilcoxon may fail
            stat, p = np.nan, 1.0
        rows.append({
            "comparison": f"{s1}→{s2}",
            "n_pairs": len(common),
            "mean_diff": diff.mean(),
            "median_diff": diff.median(),
            "wilcoxon_stat": stat,
            "p_value": p
        })
    return pd.DataFrame(rows)

# =========================
# 3) Suggestions effect inside the subset (optional)
# =========================
def suggestions_effect(fe_df, team_ids, leader,
                       team_col="team_id", leader_col="leader_type",
                       perf_col="performance", sugg_col="suggestions"):
    if sugg_col not in fe_df.columns:
        return None, None
    sub = fe_df[(fe_df[team_col].isin(team_ids)) & (fe_df[leader_col]==leader)]
    if sub[sugg_col].dropna().nunique() < 2:
        return None, None  # not enough variation
    # Mean performance by suggestions
    agg = (sub.groupby(sugg_col)[perf_col]
             .agg(["mean","median","std","count"]).reset_index()
             .rename(columns={sugg_col:"suggestions"}))
    # Simple OLS with suggestions and session controls
    model = smf.ols(f"{perf_col} ~ C({sugg_col}) + { 'session' if 'session' in sub.columns else '1' }", data=sub).fit()
    return agg, model

# =========================
# 4) Plot helpers
# =========================
def plot_trajectories(df, title, session_col="session", perf_col="performance", group=None):
    plt.figure(figsize=(8,5))
    # Average with CI-style error bars (SE)
    summ = (df.groupby(session_col)[perf_col]
              .agg(["mean","std","count"]).reset_index())
    se = summ["std"]/np.sqrt(summ["count"].clip(lower=1))
    plt.errorbar(summ[session_col], summ["mean"], yerr=se, marker="o", linestyle="-", capsize=3)
    # Optional: faint team lines
    for tid, dd in df.groupby("team_id"):
        plt.plot(dd[session_col], dd[perf_col], alpha=0.2)
    plt.title(title)
    plt.xlabel("Session")
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import itertools
import re
from typing import Dict, Optional, Tuple, List
from scipy.stats import spearmanr, norm
from statsmodels.stats.multitest import multipletests

# ---------- utilities ----------
def _sanitize_label_token(s: str) -> str:
    return re.sub(r'[^A-Za-z0-9_]+', '_', str(s)).strip('_')



def fisher_pdiff(rA: float, nA: int, rB: float, nB: int) -> Tuple[float, float]:
    """Return (z_stat, p_two_sided) for difference between two independent correlations."""
    zA = fisher_z(rA); zB = fisher_z(rB)
    se = np.sqrt(1/(max(nA, 4) - 3) + 1/(max(nB, 4) - 3))  # guard tiny n
    z = (zA - zB) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p

# ---------- per-layout correlations (session level, no deltas) ----------
def correlations_per_layout_no_transitions_with_np(
    df_sessions: pd.DataFrame,
    layout_col: str = 'layout',
    target_col: str = 'performance',
    exclude_patterns = (
        r"^dangerous_?weighted_?distance$",
        r"^dangerous_?weithed_?distance$",  # common typo
    ),
    abs_threshold: float = None,  # filter for per-layout tables (not required for stats)
    min_rows: int = 2,
) -> Dict[str, pd.DataFrame]:
    """
    Returns: {layout_value: DataFrame[feature, r, p, n]}
    """
    if target_col not in df_sessions.columns:
        raise ValueError(f"Column '{target_col}' not found.")
    if layout_col not in df_sessions.columns:
        raise ValueError(f"Layout column '{layout_col}' not found.")

    feat_df = df_sessions.copy().replace([np.inf, -np.inf], np.nan)
    if not pd.api.types.is_numeric_dtype(feat_df[target_col]):
        feat_df[target_col] = pd.to_numeric(feat_df[target_col], errors='coerce')

    id_like = {layout_col, 'team_id', 'session', 'leader_type', target_col}  # <-- exclude target here
    numeric_feats = [c for c in feat_df.columns
                     if c not in id_like and pd.api.types.is_numeric_dtype(feat_df[c])]

    excl = re.compile("|".join(exclude_patterns), flags=re.IGNORECASE)
    numeric_feats = [c for c in numeric_feats if not excl.search(c)]

    out: Dict[str, pd.DataFrame] = {}
    for lv, sub in feat_df.groupby(layout_col):
        rows: List[dict] = []
        # skip super-small layouts entirely
        if len(sub) < min_rows or sub[target_col].nunique(dropna=True) <= 1:
            out[lv] = pd.DataFrame(columns=['feature','r','p','n'])
            continue

        for c in numeric_feats:
            if c == target_col:        # <-- extra guard
                continue
            if c not in sub.columns: 
                continue
            x = sub[[c, target_col]].dropna()
            
            if x.columns.duplicated().any():     # <-- defensive: handles accidental dups
                x = x.loc[:, ~x.columns.duplicated()]
            # ensure both columns still present and have variation
            if c not in x.columns or target_col not in x.columns:
                continue
            # need variation in both
            if x[c].nunique() <= 1 or x[target_col].nunique() <= 1 or len(x) < min_rows:
                continue
            r, p = spearmanr(x[c], x[target_col])
            rows.append({'feature': c, 'r': r, 'p': p, 'n': len(x)})

        df_corr = pd.DataFrame(rows)
        if not df_corr.empty:
            if abs_threshold is not None:
                df_corr = df_corr.loc[df_corr['r'].abs() >= abs_threshold].reset_index(drop=True)
            df_corr = df_corr.sort_values('r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
        out[lv] = df_corr

    return out

# ---------- pairwise differences with p_diff and q_diff ----------
def pairwise_layout_diff_with_pvals(
    corr_by_layout: Dict[str, pd.DataFrame],
    *,
    min_n: int = 4,
    strength_thresh: float = 0.0,   # optional |r| filter applied BEFORE testing
    fdr_scope: str = "per_pair",    # "per_pair" or "global"
    diff_table_top_k: int = 50,
) -> pd.DataFrame:
    """
    Build a long table of pairwise comparisons with Fisher z test:
    Columns: layout_A, layout_B, feature, r_A, p_A, n_A, r_B, p_B, n_B, delta_r, z_diff, p_diff, q_diff
    """
    layouts = sorted([L for L, df in corr_by_layout.items() if not df.empty])
    # normalize each per-layout df to needed columns
    norm = {}
    for L in layouts:
        df = corr_by_layout[L].copy()
        if df.empty: 
            continue
        # detect columns
        colmap = {}
        for want, candidates in [('feature', ['feature']),
                                 ('r', ['corr_with_performance','r','rho']),
                                 ('p', ['p','pval','p_value']),
                                 ('n', ['n','N','nsamples'])]:
            for cand in candidates:
                if cand in df.columns:
                    colmap[want] = cand
                    break
        if 'feature' not in colmap or 'r' not in colmap:
            continue
        # ensure numeric types
        df[colmap['r']] = pd.to_numeric(df[colmap['r']], errors='coerce')
        if 'p' in colmap:
            df[colmap['p']] = pd.to_numeric(df[colmap['p']], errors='coerce')
        else:
            df['p'] = np.nan
            colmap['p'] = 'p'
        if 'n' in colmap:
            df[colmap['n']] = pd.to_numeric(df[colmap['n']], errors='coerce')
        else:
            df['n'] = np.nan
            colmap['n'] = 'n'
        slim = df[[colmap['feature'], colmap['r'], colmap['p'], colmap['n']]].rename(
            columns={colmap['feature']:'feature', colmap['r']:'r', colmap['p']:'p', colmap['n']:'n'}
        )
        if strength_thresh > 0:
            slim = slim.loc[slim['r'].abs() >= strength_thresh]
        norm[L] = slim.set_index('feature')

    rows = []
    all_pairs = list(itertools.combinations(sorted(norm.keys()), 2))
    for A, B in all_pairs:
        A_df = norm[A]; B_df = norm[B]
        inter = A_df.join(B_df, how='inner', lsuffix='_A', rsuffix='_B')
        if inter.empty:
            continue
        # compute per-feature Fisher p_diff (guard min_n & degenerate)
        p_diffs = []
        for feat, r in inter.iterrows():
            rA, nA = r['r_A'], r['n_A']
            rB, nB = r['r_B'], r['n_B']
            if (pd.isna(rA) or pd.isna(rB) or pd.isna(nA) or pd.isna(nB) or
                nA < min_n or nB < min_n):
                p_diff = np.nan; z_diff = np.nan
            else:
                z_diff, p_diff = fisher_pdiff(float(rA), int(nA), float(rB), int(nB))
            rows.append({
                'layout_A': A, 'layout_B': B, 'feature': feat,
                'r_A': rA, 'p_A': r['p_A'], 'n_A': nA,
                'r_B': rB, 'p_B': r['p_B'], 'n_B': nB,
                'delta_r': (rA - rB) if pd.notna(rA) and pd.notna(rB) else np.nan,
                'z_diff': z_diff, 'p_diff': p_diff
            })
            p_diffs.append(p_diff)

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df

    # FDR
    if fdr_scope == "per_pair":
        q_vals = np.full(len(pair_df), np.nan)
        for (A,B), sub_idx in pair_df.groupby(['layout_A','layout_B']).groups.items():
            mask = pair_df.loc[list(sub_idx), 'p_diff'].notna()
            if not mask.any(): 
                continue
            _, q, _, _ = multipletests(pair_df.loc[list(sub_idx), 'p_diff'][mask], method='fdr_bh')
            q_vals[list(np.array(list(sub_idx))[mask.to_numpy()])] = q
        pair_df['q_diff'] = q_vals
    else:  # global
        mask = pair_df['p_diff'].notna()
        q_vals = np.full(len(pair_df), np.nan)
        if mask.any():
            _, q, _, _ = multipletests(pair_df.loc[mask, 'p_diff'], method='fdr_bh')
            q_vals[mask.to_numpy()] = q
        pair_df['q_diff'] = q_vals

    # nice ordering helper cols
    pair_df['abs_delta_r'] = pair_df['delta_r'].abs()
    pair_df = pair_df.sort_values(['layout_A','layout_B','abs_delta_r'], ascending=[True,True,False]).reset_index(drop=True)
    return pair_df

# ---------- LaTeX writer (pairwise tables now include p_diff/q_diff) ----------
def write_per_layout_latex_with_meta_and_stats(
    corr_by_layout: Dict[str, pd.DataFrame],
    out_tex: str = "layout_session_corr_tables.tex",
    note_excluded: str = "excluding dangerous_weighted_distance",
    # per-layout display
    abs_thresh: Optional[float] = None,
    top_k: int = 10,
    require_all_layouts: bool = True,
    min_layouts: Optional[int] = None,
    pretty_layout_names: Optional[Dict[str,str]] = None,
    # pairwise/statistics knobs
    do_pairwise: bool = True,
    strength_thresh: float = 0.0,         # prefilter |r| for pairwise testing
    min_n: int = 4,
    fdr_scope: str = "per_pair",          # "per_pair" or "global"
    top_k_pair: int = 50,
):
    # 1) Per-layout simple tables (reuse what you already had, now we expect columns ['feature','r','p','n'])
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: session-level correlations with performance per layout\n\n")
        for lv, tbl in corr_by_layout.items():
            name = pretty_layout_names.get(lv, str(lv)) if pretty_layout_names else str(lv)
            safe = _sanitize_label_token(name)
            df = tbl.copy()
            if df.empty:
                empty = pd.DataFrame({'feature': [], 'r': [], 'p': [], 'n': []})
                f.write(f"%% -------- Layout {name} --------\n")
                f.write(empty.to_latex(index=False, float_format="%.3f",
                                       caption=f"Session-level Spearman correlations with performance for layout {name} ({note_excluded})",
                                       label=f"tab:session_corr_layout_{safe}"))
                f.write("\n\n")
                continue
            if abs_thresh is not None:
                df = df.assign(abs_r=df['r'].abs()).loc[lambda d: d['abs_r']>=abs_thresh].drop(columns='abs_r')
            df = df.sort_values('r', key=lambda s: s.abs(), ascending=False).head(top_k)
            f.write(f"%% -------- Layout {name} --------\n")
            f.write(df[['feature','r','p','n']].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=f"Session-level Spearman correlations with performance for layout {name} ({note_excluded})",
                label=f"tab:session_corr_layout_{safe}"
            ))
            f.write("\n\n")

    # 2) Cross-layout meta (shared/dissimilar across all layouts), same as before — optional; omitted for brevity.

    # 3) Pairwise with p_diff/q_diff
    if not do_pairwise:
        print(f"Wrote LaTeX tables to: {out_tex}")
        return

    pair_df = pairwise_layout_diff_with_pvals(
        corr_by_layout, min_n=min_n, strength_thresh=strength_thresh, fdr_scope=fdr_scope, diff_table_top_k=top_k_pair
    )
    with open(out_tex, "a") as f:
        f.write("%% -------- PAIRWISE COMPARISONS BETWEEN LAYOUTS (with p_diff and q_diff) --------\n\n")

    if pair_df.empty:
        print(f"Wrote LaTeX tables to: {out_tex} (no pairwise rows)")
        return

    # For each (A,B), write two tables ranked by |Δr|
    for (A,B), sub in pair_df.groupby(['layout_A','layout_B']):
        nameA = A; nameB = B
        safeA = _sanitize_label_token(nameA); safeB = _sanitize_label_token(nameB)
        sub = sub.copy()
        # Shared = same sign; Dissimilar = opposite sign
        sub['same_sign'] = np.sign(sub['r_A']) == np.sign(sub['r_B'])
        shared = sub[sub['same_sign']].sort_values('abs_delta_r', ascending=True).head(top_k_pair)
        dissim = sub[~sub['same_sign']].sort_values('abs_delta_r', ascending=False).head(top_k_pair)

        cols = ['feature','r_A','p_A','n_A','r_B','p_B','n_B','delta_r','z_diff','p_diff','q_diff']
        with open(out_tex, "a") as f:
            f.write(f"%% -------- PAIR: {nameA} vs {nameB} — Shared (same sign) --------\n")
            f.write(shared[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {nameA} vs {nameB}: shared-direction correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:layout_pair_shared_{safeA}_vs_{safeB}"
            ))
            f.write("\n\n")

            f.write(f"%% -------- PAIR: {nameA} vs {nameB} — Dissimilar (opposite sign) --------\n")
            f.write(dissim[cols].round(3).to_latex(
                index=False, float_format="%.3f",
                caption=(f"Layouts {nameA} vs {nameB}: opposite-direction correlations with Fisher test for difference "
                         f"(p_diff, q_diff; {note_excluded})."),
                label=f"tab:layout_pair_dissimilar_{safeA}_vs_{safeB}"
            ))
            f.write("\n\n")

    print(f"Wrote LaTeX tables (with p_diff & q_diff) to: {out_tex}")





def per_team_performance_analysis_with_globals(
    df_sessions: pd.DataFrame,
    team_col: str = "team_id",
    leader_col: str = "leader_type",
    perf_col: str = "performance",
    suggestions_col: Optional[str] = "suggestions",   # "yes"/"no" expected if provided
    include_features: Optional[list] = None,          # None = auto-detect numeric non-id columns
    exclude_cols: Tuple[str, ...] = ("team_id","session","leader_type","layout","name"),
    min_rows_corr: int = 3,                           # minimum rows to compute any correlation
    do_fisher_within_team: bool = True,               # Fisher test AI vs Human within a team (uses stratum="all")
    fdr_scope: str = "team",                          # "team" or "none" (BH within team for Fisher’s tests)
    z_scope: str = "global",                          # "global" | "group": z-scoring against all rows vs (leader,suggestions) group
    suggestions_mode: str = "binary",                 # keep "binary" for yes/no
    fallback_attr: bool = True                        # per-team attribution using global/group r if correlations are impossible
) -> Dict[str, any]:
    """
    Adds global comparisons:
      - Global Spearman correlations per (leader) and per (leader × suggestions yes/no).
      - Per-team attribution uses r from the matching global group.
      - Optional Fisher tests AI vs Human within team (uses stratum='all').

    Returns:
      {
        "by_team_leader": { (team, leader): DataFrame[feature, r, p, n, stratum] }   # correlations (rare if n>=3)
        "fisher_within_team": { team: DataFrame[feature, r_human, n_human, r_ai, n_ai, z_diff, p_diff, q_diff] }
        "fallback_attr": { (team, leader, sugg): DataFrame[feature, z, r_global, score, group_key] }
        "global_corr": { group_key: DataFrame[feature, r, p, n] }                    # group_key = (leader, sugg) with sugg in {"all","yes","no"}
        "feature_list": [ ... ]                                                      # the features analyzed
      }
    """
    df = df_sessions.copy()

    # --- Normalize labels ---
    if leader_col not in df.columns:
        raise ValueError(f"'{leader_col}' not found.")
    if perf_col not in df.columns:
        raise ValueError(f"'{perf_col}' not found.")
    df[perf_col] = pd.to_numeric(df[perf_col], errors="coerce")

    df[leader_col] = (df[leader_col].astype(str)
                      .str.strip().str.lower()
                      .map(lambda s: "human" if "human" in s else ("ai" if "ai" in s else s)))

    if suggestions_col and suggestions_col in df.columns and suggestions_mode == "binary":
        df[suggestions_col] = (df[suggestions_col].astype(str).str.strip().str.lower()
                               .map({"yes":"yes","y":"yes","true":"yes","1":"yes",
                                     "no":"no","n":"no","false":"no","0":"no"})
                               .fillna(df[suggestions_col]))

    # --- Detect numeric features (exclude IDs, perf, suggestions) ---
    if include_features is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        excluded = set(exclude_cols) | {perf_col}
        if suggestions_col is not None:
            excluded |= {suggestions_col}
        features = [c for c in num_cols if c not in excluded]
    else:
        features = include_features

    if not features:
        raise ValueError("No candidate features to analyze. Provide `include_features` or verify numeric columns.")

    # --- Helper: correlation table over a subset ---
    def _corr_table(sub: pd.DataFrame) -> pd.DataFrame:
        y = sub[perf_col]
        if y.nunique(dropna=True) < 2:
            return pd.DataFrame(columns=["feature","r","p","n"])
        rows = []
        for f in features:
            x = sub[f]
            if x.nunique(dropna=True) < 2:
                continue
            xy = pd.concat([x, y], axis=1).dropna()
            if xy.shape[0] < min_rows_corr:
                continue
            r, p = spearmanr(xy.iloc[:,0], xy.iloc[:,1])
            rows.append({"feature": f, "r": float(r), "p": float(p), "n": int(xy.shape[0])})
        out = pd.DataFrame(rows)
        if out.empty:
            return out
        return out.sort_values("r", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    # --- Helper: Fisher z-test for two independent correlations ---
    def _fisher_pdiff(rA: float, nA: int, rB: float, nB: int) -> Tuple[float, float]:
        if min(nA, nB) < 4:
            return (np.nan, np.nan)
        rA = float(np.clip(rA, -0.999999, 0.999999))
        rB = float(np.clip(rB, -0.999999, 0.999999))
        zA = 0.5 * np.log((1 + rA) / (1 - rA))
        zB = 0.5 * np.log((1 + rB) / (1 - rB))
        se = np.sqrt(1/(nA - 3) + 1/(nB - 3))
        z = (zA - zB) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        return z, p

    # --- GLOBAL correlations by group ---
    # Keys: (leader, sugg) where sugg in {"all","yes","no"}
    groups = []
    for lt in ["ai","human"]:
        groups.append((lt, "all"))
        if suggestions_col and suggestions_mode == "binary" and suggestions_col in df.columns:
            groups.append((lt, "yes"))
            groups.append((lt, "no"))

    global_corr: Dict[Tuple[str,str], pd.DataFrame] = {}
    for (lt, sg) in groups:
        if sg == "all":
            sub = df[df[leader_col] == lt]
        else:
            sub = df[(df[leader_col] == lt) & (df[suggestions_col] == sg)]
        global_corr[(lt, sg)] = _corr_table(sub)

    # --- z-score baselines (for attribution) ---
    # We can z against the whole dataset (global) OR within each (leader,sugg) group.
    if z_scope == "global":
        z_means = df[features].mean()
        z_stds  = df[features].std().replace(0, np.nan)
        z_baseline_map = {("all","all"): (z_means, z_stds)}  # sentinel
    elif z_scope == "group":
        z_baseline_map = {}
        for (lt, sg) in groups:
            if sg == "all":
                sub = df[df[leader_col] == lt]
            else:
                sub = df[(df[leader_col] == lt) & (df[suggestions_col] == sg)]
            mu = sub[features].mean()
            sd = sub[features].std().replace(0, np.nan)
            z_baseline_map[(lt, sg)] = (mu, sd)
    else:
        raise ValueError("z_scope must be 'global' or 'group'.")

    # --- per (team, leader) correlations (likely empty given 1-row issue), and fallback attribution per (team, leader, sugg) ---
    by_team_leader: Dict[Tuple, pd.DataFrame] = {}
    fallback_attr: Dict[Tuple, pd.DataFrame] = {}

    # Pre-compute the per-(team,leader,sugg) rows
    if suggestions_col and suggestions_col in df.columns:
        triplet_groups = df.groupby([team_col, leader_col, suggestions_col], dropna=False)
    else:
        # fall back to (team, leader) only
        triplet_groups = df.groupby([team_col, leader_col], dropna=False)

    # Also compute (team, leader) tables with stratum='all' for Fisher comparisons
    pair_groups = df.groupby([team_col, leader_col], dropna=False)

    # 1) Attempt within-(team,leader) correlations (stratum='all')
    for (tid, lt), g in pair_groups:
        g = g.copy()
        # require >= min_rows_corr for correlations (usually false per your data)
        if g.shape[0] >= min_rows_corr:
            tbl = _corr_table(g)
            if not tbl.empty:
                tbl = tbl.assign(stratum="all")
                by_team_leader[(tid, lt)] = tbl

    # 2) Fallback attribution PER (team, leader, suggestions_level)
    for key, g in triplet_groups:
        g = g.copy()
        if suggestions_col and suggestions_col in df.columns:
            if len(key) == 3:
                tid, lt, sg = key
                sg = "yes" if str(sg).lower()=="yes" else ("no" if str(sg).lower()=="no" else str(sg))
            else:
                # should not happen, but guard
                tid, lt = key
                sg = "all"
        else:
            tid, lt = key
            sg = "all"

        # choose r-map from global correlations of the *matching* group
        # fallback to (lt, "all") if (lt, sg) is empty
        r_tbl = global_corr.get((lt, sg))
        if r_tbl is None or r_tbl.empty:
            r_tbl = global_corr.get((lt, "all"), pd.DataFrame(columns=["feature","r"]))
        r_map = dict(zip(r_tbl["feature"], r_tbl["r"])) if not r_tbl.empty else {}

        # choose z-baseline
        if z_scope == "global":
            mu, sd = z_baseline_map[("all","all")]
        else:
            mu, sd = z_baseline_map.get((lt, sg), (df[features].mean(), df[features].std().replace(0, np.nan)))

        # the single row (or few rows) for this team-condition: use mean across rows (usually 1)
        xbar = g[features].mean()
        z = (xbar - mu) / sd
        z = z.replace([np.inf, -np.inf], np.nan)

        r_vec = pd.Series([r_map.get(f, 0.0) for f in features], index=features)
        score = z * r_vec * r_vec.abs()

        out = (pd.DataFrame({"feature": features, "z": z.values, "r_global": r_vec.values, "score": score.values})
                 .dropna(subset=["z"])
                 .sort_values("score", ascending=False)
                 .reset_index(drop=True))
        out.insert(0, "group_key", [(lt, sg)] * len(out))
        fallback_attr[(tid, lt, sg)] = out

    # 3) Fisher tests AI vs Human within-team (uses stratum='all' correlations if they exist)
    fisher_within_team: Dict[any, pd.DataFrame] = {}
    if do_fisher_within_team:
        # build 'all' stratum on-the-fly if missing
        # (with your 1-row cells this will likely be absent; that's fine)
        for tid in df[team_col].unique():
            h = by_team_leader.get((tid, "human"))
            a = by_team_leader.get((tid, "ai"))
            if h is None or a is None or h.empty or a.empty:
                continue
            H = h[["feature","r","n"]].rename(columns={"r":"r_human","n":"n_human"})
            A = a[["feature","r","n"]].rename(columns={"r":"r_ai","n":"n_ai"})
            m = H.merge(A, on="feature", how="inner")
            if m.empty: 
                continue
            zs, ps = [], []
            for row in m.itertuples(index=False):
                z, p = _fisher_pdiff(row.r_human, int(row.n_human), row.r_ai, int(row.n_ai))
                zs.append(z); ps.append(p)
            m["z_diff"] = zs
            m["p_diff"] = ps

            if fdr_scope == "team":
                mask = m["p_diff"].notna()
                qvals = np.full(m.shape[0], np.nan)
                if mask.any():
                    _, q, _, _ = multipletests(m.loc[mask, "p_diff"], method="fdr_bh")
                    qvals[mask.values] = q
                m["q_diff"] = qvals

            fisher_within_team[tid] = m.sort_values("p_diff", na_position="last").reset_index(drop=True)

    return {
        "by_team_leader": by_team_leader,
        "fisher_within_team": fisher_within_team,
        "fallback_attr": fallback_attr,          # key = (team, leader, suggestions)
        "global_corr": global_corr,              # key = (leader, sugg in {"all","yes","no"})
        "feature_list": features,
    }


# ---------------------- Pretty Printers ----------------------

def print_global_drivers(res, top_k: int = 12):
    """Show the top-|r| global drivers for each (leader, suggestions) group."""
    print("=== GLOBAL DRIVERS (Spearman |r|) ===")
    for key in sorted(res["global_corr"].keys()):
        lt, sg = key
        tbl = res["global_corr"][key]
        if tbl is None or tbl.empty:
            continue
        show = (tbl.assign(abs_r=tbl["r"].abs())
                    .sort_values("abs_r", ascending=False)
                    .head(top_k)
                    .drop(columns=["abs_r"]))
        print(f"\n-- Leader: {lt.upper()} | Suggestions: {sg.upper()} --  (n varies by feature)")
        print(f"{'feature':<40} {'r':>7} {'p':>8} {'n':>5}")
        for r in show.itertuples(index=False):
            print(f"{r.feature:<40} {r.r:>7.3f} {r.p:>8.3f} {int(r.n):>5}")


def print_team_vs_global(res, df_sessions, team_col="team_id", leader_col="leader_type",
                         suggestions_col="suggestions", top_k: int = 10):
    """
    For each available (team, leader, suggestions) attribution table, show how
    the team's z aligns with the group's global r.
    """
    print("\n\n=== TEAM vs GLOBAL (Top drivers by score; show z and global r sign) ===")
    # normalize suggestions for labeling
    if suggestions_col in df_sessions.columns:
        sugg_norm = (df_sessions[suggestions_col].astype(str).str.strip().str.lower()
                     .map({"yes":"yes","y":"yes","true":"yes","1":"yes",
                           "no":"no","n":"no","false":"no","0":"no"}))
    else:
        sugg_norm = pd.Series(index=df_sessions.index, dtype=object)

    # known triplets in result
    keys = sorted(res["fallback_attr"].keys(),
                  key=lambda k: (int(k[0]) if str(k[0]).isdigit() else k[0], k[1], str(k[2])))

    for (tid, lt, sg) in keys:
        tbl = res["fallback_attr"][(tid, lt, sg)]
        if tbl is None or tbl.empty:
            continue
        print("\n" + "-"*88)
        print(f"Team {tid} | Leader: {lt} | Suggestions: {sg}")
        show = tbl.head(top_k).copy()
        # alignment indicator: sign(z) * sign(r_global)
        align = np.sign(show["z"].fillna(0)) * np.sign(show["r_global"].fillna(0))
        show = show.assign(align=align.map({1.0:"+", -1.0:"−", 0.0:"0"}))
        print(f"{'feature':<40} {'z':>7} {'r_gl':>7} {'align':>7} {'score':>9}")
        for r in show.itertuples(index=False):
            print(f"{r.feature:<40} {r.z:>7.3f} {r.r_global:>7.3f} {r.align:>7} {r.score:>9.3f}")



# -----------------------------------------
# Figure 1: correlation trajectories (lines)
# -----------------------------------------
def plot_correlations_over_sessions(session_corr_df: pd.DataFrame,
                                    features: list,
                                    by: list,
                                    out_png: str = "corr_trajectories.png",
                                    title: str = "Correlation with performance across sessions"):
    # ensure all needed sessions exist; missing -> NaN will just break the line
    sess_vals = sorted(session_corr_df[by].dropna().unique().tolist())

    cmap = cm.get_cmap("tab20", len(features))
    plt.figure(figsize=(8, 5))
    for idx,feat in enumerate(features):
        sub = session_corr_df.loc[session_corr_df['feature'] == feat, [by, 'corr']]
        sub = sub.set_index(by).reindex(sess_vals)  # keep order 0..3
        # Basic line; no manual color set (let matplotlib handle defaults)
        plt.plot(sess_vals, sub['corr'].values, marker='o', label=feat, color=cmap(idx % cmap.N))

    plt.axhline(0, linewidth=1, linestyle='--')
    plt.xlabel(by)
    plt.ylabel("Spearman r (feature ~ performance)")
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved: {out_png}")


def plot_correlations_by_leader_suggestions(
    corr_df: pd.DataFrame,
    features: list,
    leader_col: str = "leader_type",         # e.g., {'human','ai'}
    sugg_col: str = "suggestions",           # e.g., {0/1 or False/True or 'without'/'with'}
    corr_col: str = "corr",                  # Spearman r (feature ~ performance)
    feature_col: str = "feature",
    out_png: str = "corr_leader_sugg_trajectories.png",
    title: str = "Correlation with performance across leader × suggestions",
    category_order = None,      # optional explicit order for x-axis categories
    labels_map = None           # optional pretty labels for x-ticks
):
    """
    Expect a long-form dataframe with columns:
      - feature (feature_col)
      - corr (corr_col)
      - leader_type (leader_col)
      - suggestions (sugg_col)
    One row per (feature, leader_type, suggestions) combo.
    """

    df = corr_df.copy()

    # Normalize suggestions to a readable, consistent form for labeling
    # (supports booleans, 0/1, or strings)
    def _norm_sugg(v):
        if isinstance(v, (int, np.integer)):
            return "with_sugg" if int(v) == 1 else "no_sugg"
        if isinstance(v, bool):
            return "with_sugg" if v else "no_sugg"
        s = str(v).strip().lower()
        if s in {"1", "true", "yes", "y", "with", "with_suggestions", "with-suggestions"}:
            return "with_sugg"
        if s in {"0", "false", "no", "n", "without", "no_suggestions", "no-suggestions"}:
            return "no_sugg"
        # fallback: keep original, but make it filename-friendly
        return s.replace(" ", "_")

    df["_sugg_norm"] = df[sugg_col].apply(_norm_sugg)
    df["_combo"] = list(zip(df[leader_col].astype(str), df["_sugg_norm"].astype(str)))

    # Default category order if none provided
    if category_order is None:
        # Prefer a consistent, readable order
        # [('human','no_sugg'), ('human','with_sugg'), ('ai','no_sugg'), ('ai','with_sugg')]
        unique_leaders = ["human", "ai"]
        # preserve any unexpected leader values too
        extra_leaders = [l for l in df[leader_col].unique().tolist() if l not in unique_leaders]
        unique_leaders = unique_leaders + [l for l in extra_leaders if l not in unique_leaders]

        unique_sugg = ["no_sugg", "with_sugg"]
        extra_sugg = [s for s in df["_sugg_norm"].unique().tolist() if s not in unique_sugg]
        unique_sugg = unique_sugg + [s for s in extra_sugg if s not in unique_sugg]

        category_order = [(l, s) for l in unique_leaders for s in unique_sugg]

    # Build x-tick labels
    if labels_map is None:
        # default pretty labels
        def _lbl(tup):
            l, s = tup
            l_lab = "Human" if str(l).lower() == "human" else ("AI" if str(l).lower() == "ai" else str(l))
            s_lab = "w/ sugg" if s == "with_sugg" else ("no sugg" if s == "no_sugg" else s)
            return f"{l_lab}\n({s_lab})"
        xtick_labels = [_lbl(t) for t in category_order]
    else:
        xtick_labels = [labels_map.get(t, f"{t[0]}\n({t[1]})") for t in category_order]

    x_vals = np.arange(len(category_order))

    # Plot
    cmap = cm.get_cmap("tab20", max(1, len(features)))
    plt.figure(figsize=(9, 5.5))
    for i, feat in enumerate(features):
        sub = df.loc[df[feature_col] == feat, ["_combo", corr_col]].copy()
        # reindex by category_order to ensure consistent x order
        sub = sub.set_index("_combo").reindex(category_order)
        y = sub[corr_col].values.astype(float)

        # Draw the line (allow NaNs to break segments)
        plt.plot(x_vals, y, marker="o", label=feat, color=cmap(i % cmap.N))

    # Zero line
    plt.axhline(0, linewidth=1, linestyle="--")

    plt.xticks(x_vals, xtick_labels)
    plt.xlabel("Leader × Suggestions")
    plt.ylabel("Spearman r (feature ~ performance)")
    plt.title(title)
    plt.legend(loc="best", fontsize=8, ncols=1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()
    print(f"Saved: {out_png}")


def plot_sankey_style_correlations(
    corr_long: pd.DataFrame,
    top_k: int = 20,
    min_sessions: int = 2,
    features = None,
    figsize=(12, 7),
    lw_min: float = 1.0,
    lw_max: float = 10.0,
    annotate_r: bool = True,
    jitter: float = 0.0,
):
    """
    Draw an 'alluvial / sankey-style' slope plot that links the SAME features across sessions.

    Parameters
    ----------
    corr_long : pd.DataFrame
        Must have columns ['session','feature','r'].
    top_k : int
        If `features` is None, pick the top_k by mean |r| across sessions.
    min_sessions : int
        Keep only features that appear in at least this many sessions.
    features : list[str] | None
        Explicit list of features to plot. Overrides top_k if provided.
    """

    # --- validate input
    required_cols = {"session", "feature", "corr"}
    missing = required_cols - set(corr_long.columns)
    if missing:
        raise ValueError(f"corr_long missing required columns: {sorted(missing)}")

    df = corr_long.copy()
    df["corr"] = pd.to_numeric(df["corr"], errors="coerce")
    df = df.dropna(subset=["corr"])

    # aggregate if duplicates
    df = (df.groupby(["session", "feature"], as_index=False)["corr"]
            .mean())

    # session order
    if pd.api.types.is_categorical_dtype(df["session"]):
        sessions = list(df["session"].cat.categories)
    else:
        sessions = sorted(df["session"].unique())

    # filter by min_sessions
    counts = df.groupby("feature")["session"].nunique()
    keep_feats = counts[counts >= min_sessions].index

    # either use explicit features or rank by mean |r|
    if features is not None:
        chosen = [f for f in features if f in keep_feats]
    else:
        mean_abs = (df[df["feature"].isin(keep_feats)]
                      .assign(abs_r=lambda x: x["corr"].abs())
                      .groupby("feature")["abs_r"].mean()
                      .sort_values(ascending=False))
        chosen = list(mean_abs.head(top_k).index)

    df = df[df["feature"].isin(chosen)].copy()
    if df.empty:
        raise ValueError("No features left to plot after filtering")

    # feature order
    mean_abs = (df.assign(abs_r=lambda x: x["corr"].abs())
                  .groupby("feature")["abs_r"].mean()
                  .sort_values(ascending=False))
    feat_order = mean_abs.index.tolist()
    y_map = {feat: i for i, feat in enumerate(feat_order[::-1], start=1)}

    # correlation lookup
    r_map = {(row.session, row.feature): row.corr for row in df.itertuples(index=False)}

    abs_vals = df["corr"].abs().values
    r_min, r_max = float(np.nanmin(abs_vals)), float(np.nanmax(abs_vals))
    if np.isclose(r_min, r_max):
        to_lw = lambda r: (lw_min + lw_max) * 0.5
    else:
        to_lw = lambda r: lw_min + (lw_max - lw_min) * ((abs(r) - r_min) / (r_max - r_min))

    fig, ax = plt.subplots(figsize=figsize)
    x_pos = {s: i for i, s in enumerate(sessions)}

    # vertical guides
    for s in sessions:
        ax.axvline(x_pos[s], alpha=0.2)

    for feat in feat_order:
        y = y_map[feat]
        if jitter:
            y = y + np.random.uniform(-jitter, jitter)
        r_per_sess = [r_map.get((s, feat), np.nan) for s in sessions]

        for i in range(len(sessions) - 1):
            r_left, r_right = r_per_sess[i], r_per_sess[i + 1]
            if np.isnan(r_left) or np.isnan(r_right):
                continue
            xs0, xs1 = x_pos[sessions[i]], x_pos[sessions[i + 1]]
            r_mean = 0.5 * (r_left + r_right)
            linestyle = "-" if r_mean >= 0 else ":"
            lw = to_lw(r_mean)
            ax.plot([xs0, xs1], [y, y], linestyle=linestyle, linewidth=lw, alpha=0.9)

        if annotate_r:
            for s, r_val in zip(sessions, r_per_sess):
                if np.isnan(r_val):
                    continue
                ax.text(x_pos[s], y, f"{r_val:+.2f}",
                        va="center", ha="center", fontsize=8, alpha=0.9)

    ax.set_xticks([x_pos[s] for s in sessions])
    ax.set_xticklabels([f"Session {s}" for s in sessions])
    ax.set_xlim(min(x_pos.values()) - 0.5, max(x_pos.values()) + 0.5)

    ax.set_yticks([y_map[f] for f in feat_order])
    ax.set_yticklabels(feat_order)
    ax.invert_yaxis()

    style_handles = [
        Line2D([0], [0], linestyle="-", linewidth=2, label="Positive r"),
        Line2D([0], [0], linestyle=":", linewidth=2, label="Negative r"),
    ]
    ex_r = np.linspace(r_min, r_max, 3)
    width_handles = [
        Line2D([0], [0], linestyle="-", linewidth=to_lw(rv), label=f"|r|≈{abs(rv):.2f}")
        for rv in ex_r
    ]
    ax.legend(handles=style_handles + width_handles, loc="upper right",
              frameon=False, title="Encoding")

    ax.set_title("Feature ↔ Performance correlations across sessions")
    ax.set_ylabel("Features")
    plt.tight_layout()
    return fig, ax

def plot_corr_heatmap(
    corr_long: pd.DataFrame,
    features=None,
    min_sessions: int = 1,
    session_order=None,
    annotate: bool = True,
    fmt: str = ".2f",
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "coolwarm",
    by=None,
    figsize=(10, 8),
    show_nan_glyph: bool = True,
    force_columns = None,
    feature_label_map = None,   # <-- NEW
):
    """
    Heatmap of feature ↔ performance correlations across groups (or sessions).
    corr_long must have columns [by, 'feature', 'corr'].
    feature_label_map: maps *prefix-stripped* labels to display labels.
                       If original label startswith 'leader_' or 'nonleader_',
                       we strip that before lookup, then append ' 1' (leader) or ' 2' (nonleader).
    """

    feature_label_map = translation
    required = {by, "feature", "corr"}
    missing = required - set(corr_long.columns)
    if missing:
        raise ValueError(f"corr_long missing required columns: {sorted(missing)}")

    df = corr_long.copy()
    df["corr"] = pd.to_numeric(df["corr"], errors="coerce")  # keep NaNs

    # Decide feature universe
    if features is not None:
        df = df[df["feature"].isin(features)]
        base_for_counts = corr_long[corr_long["feature"].isin(features)]
    else:
        base_for_counts = corr_long

    # Keep features that appear in >= min_sessions groups
    counts = (base_for_counts[[by, "feature"]]
              .drop_duplicates()
              .groupby("feature")[by].nunique())
    keep_feats = counts[counts >= min_sessions].index
    df = df[df["feature"].isin(keep_feats)]

    if df.empty and features is not None:
        df = pd.DataFrame({by: [], "feature": [], "corr": []})

    # Decide column order
    df[by] = df[by].astype(str).str.strip()
    if session_order is None:
        try:
            sess_vals = sorted({int(s) for s in df[by] if str(s).isdigit()})
            if not sess_vals:
                raise ValueError
            session_order = [str(s) for s in sess_vals]
        except Exception:
            session_order = sorted(df[by].unique(), key=str)

    # Force extra columns
    if force_columns:
        force_columns = [str(c).strip() for c in force_columns]
        session_order = [str(s).strip() for s in session_order]
        for c in force_columns:
            if c not in session_order:
                session_order.append(c)

    # Pivot
    mat = (df.pivot_table(index="feature", columns=by, values="corr", aggfunc="mean")
             .reindex(columns=session_order))

    # Warn if some requested columns are entirely NaN
    empty_cols = [c for c in mat.columns if mat[c].notna().sum() == 0]
    if empty_cols:
        print("Heatmap warning — columns with no data:", empty_cols)

    # Order features by mean |corr|
    if not mat.empty:
        strength = mat.abs().mean(axis=1).sort_values(ascending=False)
        mat = mat.loc[strength.index]

    # --- NEW: translate y-axis labels with prefix handling ---
    def _translate_label(lbl: str) -> str:
        if feature_label_map is None:
            return lbl
        tag = ""
        core = lbl
        if core.startswith("leader_"):
            core = core[len("leader_"):]
            tag = "³"
        elif core.startswith("nonleader_"):
            core = core[len("nonleader_"):]
            tag = "²"
        # translate using map; default to core if not found
        disp = feature_label_map.get(core, core)
        return f"{disp}{tag}"

    # Build heatmap
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat.values, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)

    # Ticks & labels
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels([f"{s}" for s in mat.columns], rotation=0, ha="center")
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels([_translate_label(str(x)) for x in mat.index])

    # Make tick labels bigger and bold
    ax.tick_params(axis="x", labelsize=20)  # size of x labels
    ax.tick_params(axis="y", labelsize=20)  # size of y labels
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")

    

    # Minor grid
    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="w", linewidth=0.5, alpha=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Spearman correlation with performance", rotation=90, size=20, weight="bold")
    
    cbar.ax.set_yticklabels([])

    colorbar_fontsize = 18
    ticks = cbar.get_ticks()
    for i, (tick, val) in enumerate(zip(cbar.ax.get_yticks(), ticks)):
        text = f"{val:.1f}"
        if i == 0:  # bottom extreme
            cbar.ax.text(
                0.5, tick - 0.005*(ticks[-1]-ticks[0]),  # a bit below the bar
                text, ha="center", va="top",
                fontsize=colorbar_fontsize, fontweight="bold",
                transform=cbar.ax.transData
            )
        elif i == len(ticks)-1:  # top extreme
            cbar.ax.text(
                0.5, tick + 0.005*(ticks[-1]-ticks[0]),  # a bit above the bar
                text, ha="center", va="bottom",
                fontsize=colorbar_fontsize, fontweight="bold",
                transform=cbar.ax.transData
            )
        else:  # middle ticks inside the bar
            cbar.ax.text(
                0.5, tick,
                text, ha="center", va="center",
                fontsize=colorbar_fontsize, fontweight="bold",
                transform=cbar.ax.transData,
                color="black"
            )

    # Annotate values / NaNs
    rows, cols = mat.shape
    for i in range(rows):
        for j in range(cols):
            val = mat.iat[i, j]
            if pd.notna(val) and annotate:
                ax.text(
                    j, i,
                    f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=20,          # bigger font size
                    fontweight="bold",    # bold text
                )
            elif pd.isna(val) and show_nan_glyph:
                ax.text(
                    j, i, "–",
                    ha="center", va="center",
                    fontsize=12,
                    fontweight="bold",
                    alpha=0.45,
                )

    
    #ax.set_title("Feature ↔ Performance correlations by group (heatmap)")
    fig.tight_layout()
    return fig, ax


# --- helper: pooled ("ALL sessions") correlations for features you already plot ---
def _overall_corr_long(feat_df: pd.DataFrame,
                       features: list,
                       exclude_features = None) -> pd.DataFrame:
    from scipy.stats import spearmanr
    exclude_features = set(exclude_features or [])
    rows = []
    for feat in features:
        if feat in exclude_features: 
            continue
        if feat not in feat_df.columns:
            continue
        xy = feat_df[[feat, "performance"]].dropna()
        if xy.shape[0] < 3 or xy[feat].nunique() <= 1 or xy["performance"].nunique() <= 1:
            continue
        r, _p = spearmanr(xy[feat], xy["performance"])
        r, p = spearman_permutation_test(xy[feat], xy["performance"], n_resamples=50000, seed=42)
        
        rows.append({"session": "ALL", "feature": feat, "corr": float(r), "p": p})
    return pd.DataFrame(rows)


def test_ai_vs_human_per_layout(
    df: pd.DataFrame,
    performance_col: str = "performance",
    leader_col = "leader_type",      # values expected: "ai", "human"
    layout_col = "layout",
    team_col = "team_id",     # set to None if you don't have teams
    session_col = "session",  # used only for within-team aggregation
    agg: str = "median",                  # how to summarize repeats per team/leader/layout
    min_n: int = 4                        # minimum n for any test
) -> pd.DataFrame:
    """
    For each layout, compare performance between AI- and Human-led.
    - If team_col is provided and there are teams with BOTH leader types in that layout,
      runs paired tests on per-team summaries (Wilcoxon + paired t).
    - Else, runs independent-samples tests (Welch t + Mann–Whitney).
    Also returns effect sizes and BH/FDR q-values over layouts.
    """
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")
    aggfunc = np.median if agg == "median" else np.mean

    def hedges_g(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2: return np.nan
        sx, sy = x.std(ddof=1), y.std(ddof=1)
        sp = np.sqrt(((nx-1)*sx**2 + (ny-1)*sy**2) / (nx + ny - 2)) if (nx+ny-2) > 0 else np.nan
        if not np.isfinite(sp) or sp == 0: return np.nan
        d = (x.mean() - y.mean()) / sp
        J = 1 - (3 / (4*(nx + ny) - 9)) if (nx + ny) > 2 else np.nan
        return d * J if np.isfinite(J) else d

    def cliffs_delta(x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        if x.size == 0 or y.size == 0: return np.nan
        # O(n*m) – fine for modest n
        gt = sum((xi > y).sum() for xi in x)
        lt = sum((xi < y).sum() for xi in x)
        return (gt - lt) / (x.size * y.size)

    rows = []
    for layout_val, g in df.groupby(layout_col, dropna=False):
        sub = g.dropna(subset=[performance_col, leader_col]).copy()
        ai = sub.loc[sub[leader_col] == "ai", performance_col].astype(float).dropna()
        hu = sub.loc[sub[leader_col] == "human", performance_col].astype(float).dropna()
        n_ai, n_hu = len(ai), len(hu)

        # Detect paired design: same teams ran both leader types in this layout
        paired_possible = team_col is not None and team_col in sub.columns
        is_paired, n_pairs = False, 0
        pair_ai, pair_hu = None, None

        if paired_possible:
            cols = [team_col, leader_col, performance_col]
            tmp = sub[cols].copy()
            if session_col and session_col in sub.columns:
                tmp = sub[[team_col, leader_col, session_col, performance_col]].copy()
                tmp = tmp.groupby([team_col, leader_col], as_index=False)[performance_col].agg(aggfunc)
            else:
                tmp = tmp.groupby([team_col, leader_col], as_index=False)[performance_col].agg(aggfunc)
            wide = tmp.pivot(index=team_col, columns=leader_col, values=performance_col)
            if {"ai", "human"}.issubset(set(wide.columns)):
                paired = wide.dropna(subset=["ai", "human"])
                n_pairs = len(paired)
                if n_pairs >= min_n:
                    is_paired = True
                    pair_ai = paired["ai"].to_numpy(float)
                    pair_hu = paired["human"].to_numpy(float)

        if is_paired:
            # Paired tests
            diff = pair_ai - pair_hu
            mean_ai, mean_hu = float(np.mean(pair_ai)), float(np.mean(pair_hu))
            diff_mean = float(np.mean(diff))
            # Paired t
            tstat, p_t = stats.ttest_rel(pair_ai, pair_hu, nan_policy="omit")
            # Wilcoxon (robust to non-normal, handles ties with 'pratt')
            try:
                wstat, p_w = stats.wilcoxon(pair_ai, pair_hu, zero_method="pratt", alternative="two-sided")
            except ValueError:
                wstat, p_w = np.nan, np.nan
            # Cohen's dz for paired
            dz = float(diff_mean / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan

            rows.append({
                "layout": layout_val,
                "test_type": "paired",
                "n_pairs": n_pairs,
                "n_ai": n_ai,
                "n_human": n_hu,
                "mean_ai": mean_ai,
                "mean_human": mean_hu,
                "diff_mean_ai_minus_human": diff_mean,
                "p_paired_t": float(p_t),
                "p_wilcoxon": float(p_w),
                "effect_dz": dz,
                "main_p": float(p_w) if pd.notna(p_w) else float(p_t),
            })
        else:
            # Independent-samples tests (requires enough observations on each side)
            if n_ai >= min_n and n_hu >= min_n:
                mean_ai, mean_hu = float(ai.mean()), float(hu.mean())
                diff_mean = mean_ai - mean_hu
                tstat, p_welch = stats.ttest_ind(ai, hu, equal_var=False, nan_policy="omit")
                try:
                    ustat, p_mwu = stats.mannwhitneyu(ai, hu, alternative="two-sided")
                except ValueError:
                    p_mwu = np.nan
                rows.append({
                    "layout": layout_val,
                    "test_type": "independent",
                    "n_ai": n_ai,
                    "n_human": n_hu,
                    "mean_ai": mean_ai,
                    "mean_human": mean_hu,
                    "diff_mean_ai_minus_human": diff_mean,
                    "p_welch": float(p_welch),
                    "p_mannwhitney": float(p_mwu),
                    "effect_hedges_g": float(hedges_g(ai, hu)),
                    "effect_cliffs_delta": float(cliffs_delta(ai, hu)),
                    "main_p": float(p_welch),
                })
            else:
                rows.append({
                    "layout": layout_val,
                    "test_type": "insufficient",
                    "n_ai": n_ai,
                    "n_human": n_hu,
                })

    out = pd.DataFrame(rows)

    # BH/FDR across layouts on the primary p-value used for each row
    if "main_p" in out.columns:
        mask = out["main_p"].notna()
        if mask.any():
            _, qvals, _, _ = multipletests(out.loc[mask, "main_p"], method="fdr_bh")
            out.loc[mask, "q"] = qvals
            # significance stars based on q (fallback to p if q missing)
            def stars(p): return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            out["sig"] = [stars(p) if pd.notna(p) else "" for p in out.get("q", out["main_p"]).values]

    # nice ordering
    cols_pref = [
        "layout", "test_type",
        "n_ai", "n_human", "n_pairs",
        "mean_ai", "mean_human", "diff_mean_ai_minus_human",
        "p_welch", "p_mannwhitney", "p_paired_t", "p_wilcoxon",
        "effect_hedges_g", "effect_cliffs_delta", "effect_dz",
        "main_p", "q", "sig",
    ]
    return out.reindex(columns=[c for c in cols_pref if c in out.columns]).sort_values("layout")



def correlate_all_rows(df, performance_col="performance",
                       method="spearman",
                       exclude_contains=("relative_performance",),
                       min_n=6):
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feats = [c for c in num_cols if c != performance_col
             and not any(tok in c for tok in exclude_contains)]
    rows = []
    y = df[performance_col]
    for f in sorted(feats):
        sub = pd.concat([y, df[f]], axis=1).dropna()
        if len(sub) < min_n or sub[f].nunique() < 2 or sub[performance_col].nunique() < 2:
            rows.append({"feature": f, "n": len(sub), "corr": np.nan, "p": np.nan})
            continue
        if method == "spearman":
            r, p = stats.spearmanr(sub[performance_col], sub[f])
        else:
            r, p = stats.pearsonr(sub[performance_col], sub[f])
        rows.append({"feature": f, "n": len(sub), "corr": float(r), "p": float(p)})
    out = pd.DataFrame(rows)
    m = out["p"].notna()
    if m.any():
        out.loc[m, "q"] = multipletests(out.loc[m, "p"], method="fdr_bh")[1]
    else:
        out["q"] = np.nan
    out["sig"] = out["q"].apply(lambda q: "***" if q < 0.001 else "**" if q < 0.01 else "*" if q < 0.05 else "" if pd.notna(q) else "")
    return out.sort_values("corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

# ---------- Effect size helpers ----------
def cohens_d(x):
    """Cohen's d for one-sample (difference vs 0)."""
    return np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else np.nan

def cliffs_delta(x, y):
    """
    Cliff's delta for two independent samples.
    δ = (#x>y - #x<y) / (n_x * n_y)
    """
    n_x, n_y = len(x), len(y)
    greater, less = 0, 0
    for xi in x:
        greater += np.sum(xi > y)
        less += np.sum(xi < y)
    return (greater - less) / (n_x * n_y) if n_x * n_y > 0 else np.nan

# ---------- Analysis ----------
def analyze_all_transitions(df, team_col="team_id", session_col="session",
                            leader_col="leader_type", perf_col="performance"):
    df = df.sort_values([team_col, session_col])

    # Previous values per team
    df["prev_leader"] = df.groupby(team_col)[leader_col].shift(1)
    df["prev_perf"] = df.groupby(team_col)[perf_col].shift(1)

    # Transition dataframe
    trans_df = df.dropna(subset=["prev_leader", "prev_perf"]).copy()
    trans_df["transition"] = trans_df["prev_leader"] + "→" + trans_df[leader_col]
    trans_df["delta_perf"] = trans_df[perf_col] - trans_df["prev_perf"]

    # Results container
    results = {}

    # Within-transition tests
    for t in trans_df["transition"].unique():
        d = trans_df.loc[trans_df["transition"] == t, "delta_perf"].dropna()
        if len(d) > 1:
            try:
                wilc = wilcoxon(d)
            except ValueError:  # e.g. if all differences = 0
                wilc = None
            results[t] = {
                "n": len(d),
                "mean_delta": d.mean(),
                "median_delta": d.median(),
                "wilcoxon": wilc,
                "cohens_d": cohens_d(d)
            }

    # Between-group test
    groups = [trans_df.loc[trans_df["transition"] == t, "delta_perf"].dropna()
              for t in trans_df["transition"].unique()]
    results["between"] = kruskal(*groups)

    # Pairwise posthoc (nonparametric Dunn’s test)
    results["pairwise"] = sp.posthoc_dunn(
        trans_df, val_col="delta_perf", group_col="transition", p_adjust="fdr_bh"
    )

    # Pairwise Cliff’s δ
    transitions = trans_df["transition"].unique()
    cliffs = pd.DataFrame(index=transitions, columns=transitions, dtype=float)
    for i, t1 in enumerate(transitions):
        for j, t2 in enumerate(transitions):
            if i < j:
                x = trans_df.loc[trans_df["transition"] == t1, "delta_perf"].dropna()
                y = trans_df.loc[trans_df["transition"] == t2, "delta_perf"].dropna()
                cliffs.loc[t1, t2] = cliffs_delta(x, y)
                cliffs.loc[t2, t1] = -cliffs.loc[t1, t2]
            elif i == j:
                cliffs.loc[t1, t2] = 0.0
    results["cliffs_delta"] = cliffs

    return trans_df, results

# ---------- Printing ----------
def print_results(results):
    for t, res in results.items():
        if t in ("between", "pairwise", "cliffs_delta"):
            continue
        print(f"\nTransition: {t}")
        print(f"  n = {res['n']}")
        print(f"  mean Δ = {res['mean_delta']:.3f}, median Δ = {res['median_delta']:.3f}")
        if res["wilcoxon"] is not None:
            stat, p = res["wilcoxon"].statistic, res["wilcoxon"].pvalue
            print(f"  Wilcoxon signed-rank: stat={stat:.3f}, p={p:.4f}")
        else:
            print("  Wilcoxon: not applicable")
        print(f"  Cohen’s d = {res['cohens_d']:.3f}")

    print("\n== Between-group Kruskal-Wallis ==")
    stat, p = results["between"].statistic, results["between"].pvalue
    print(f"  H={stat:.3f}, p={p:.4f}")

    print("\n== Pairwise Dunn’s test (FDR adjusted p-values) ==")
    print(results["pairwise"].round(4))

    print("\n== Pairwise Cliff’s δ (effect sizes) ==")
    print(results["cliffs_delta"].round(3))
    
demographic_qs = ['Age', 'How often do you play first-person video games on average?', 'How many times have you interacted with the TeamCollab simulation environment in the past?', 'How often do you utilize Large Language Models such as ChatGPT, Claude or similar others?', 'Generally, I trust automated agents', 'Automated agents help me solve many problems', 'I think it’s a good idea to rely on automated agents for help', 'I don’t trust the information I get from automated agents', 'Automated agents are reliable', 'I rely on automated agents']

fe_df_ai = pd.read_csv('team_stats_ai.csv')



'''
tbl_all = correlate_all_rows(fe_df_ai, performance_col="performance", method="spearman")
print(tbl_all.head(30))
'''
fe_df = pd.read_csv('team_stats.csv')

pdb.set_trace()
#fe_df[(fe_df["leader_type"] == "ai") & (fe_df["suggestions"] == "Yes")]["suggestions_exchanged"].mean()

'''
demographic_condition = 'How often do you play first-person video games on average?'
std_age = fe_df.drop_duplicates(subset=["team_id", "leader_" + demographic_condition, "nonleader_" + demographic_condition])[["leader_" + demographic_condition, "nonleader_" + demographic_condition]].stack().std()
mean_age = fe_df.drop_duplicates(subset=["team_id", "leader_" + demographic_condition, "nonleader_" + demographic_condition])[["leader_" + demographic_condition, "nonleader_" + demographic_condition]].stack().mean()

print(mean_age, std_age)
pdb.set_trace()
'''

'''
# filter for AI leader
ai_df = fe_df

# list the features you want to test
features = [*nasa_questions, *survey_questions]

results = []

for feat in features:
    group_with = ai_df.loc[ai_df["leader_type"] == "ai", feat].dropna()
    group_without = ai_df.loc[ai_df["leader_type"] == "human", feat].dropna()
    
    if len(group_with) > 1 and len(group_without) > 1:
        # Welch’s t-test (doesn’t assume equal variances)
        t, p_ttest = stats.ttest_ind(group_with, group_without, equal_var=False)
        
        # Mann–Whitney U (non-parametric alternative, less assumption heavy)
        u, p_mwu = stats.mannwhitneyu(group_with, group_without, alternative="two-sided")
        
        results.append({
            "feature": feat,
            "mean_with": group_with.mean(),
            "mean_without": group_without.mean(),
            "t_stat": t,
            "p_ttest": p_ttest,
            "u_stat": u,
            "p_mwu": p_mwu,
            "n_with": len(group_with),
            "n_without": len(group_without)
        })

results_df = pd.DataFrame(results)
print(results_df.sort_values("p_ttest"))
'''

'''
trans_df, results = analyze_all_transitions(fe_df)
print_results(results)


tbl_all = correlate_all_rows(fe_df, performance_col="performance", method="spearman")
print(tbl_all.head(30))
'''
mean_ai_performance = fe_df_ai["performance"].mean()

##### Trajectory perfomance


fe_df_copy = fe_df.copy()
fe_df_copy["session"] = fe_df_copy["session"] + 1
fe_df_copy["leader_type"] = fe_df_copy["leader_type"].replace({"human": "Human", "ai": "AI"})
fe_df_copy = fe_df_copy.rename(columns={"session":"Session", "performance":"Performance", "leader_type": "Leader"})

sns.lineplot(
    data=fe_df_copy,
    x="Session", y="Performance",
    hue="Leader", ci="sd", estimator="mean"
)

# Add striped line for mean_performance_ai
plt.axhline(y=mean_ai_performance, color="red", linestyle="--", linewidth=1.5, label="AI*")
plt.xlabel("Session", fontsize=14, fontweight="bold")
plt.ylabel("Performance", fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12) 

# Force x-axis to show integers
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.legend(fontsize=12, title_fontsize=13)
#plt.title("Performance Trajectories by Leadership Style")
plt.show()


fe_df_copy = fe_df.copy()


g = fe_df_copy.groupby(["team_id","leader_type"])["performance"].mean().unstack()
order = (g.get("ai", pd.Series(index=g.index, dtype=float)) -
         g.get("human", pd.Series(index=g.index, dtype=float))).sort_values()

plt.figure()
ypos = range(len(order))

ai_color    = "orange" #sc_a.get_facecolor()[0]

# scatter points
sc_h = plt.scatter(g.loc[order.index, "human"], ypos, marker='o', label="Hᵀ")
sc_a = plt.scatter(g.loc[order.index, "ai"], ypos, color=ai_color, marker='s', label="AIᵀ")

'''
g_mean = fe_df_copy.groupby(["team_id","leader_type"])["performance"].mean().unstack()
g_std  = fe_df_copy.groupby(["team_id","leader_type"])["performance"].std().unstack()
# scatter points with error bars (eyes)
sc_h = plt.errorbar(
    g_mean.loc[order.index, "human"], ypos,
    xerr=g_std.loc[order.index, "human"],
    fmt='o', capsize=4, elinewidth=1, label="Human"
)
sc_a = plt.errorbar(
    g_mean.loc[order.index, "ai"], ypos,
    xerr=g_std.loc[order.index, "ai"],
    fmt='s', capsize=4, elinewidth=1, label="AI"
)
'''
# connecting lines
for i, t in enumerate(order.index):
    x_h = g.loc[t, "human"] if "human" in g.columns and t in g.index else np.nan
    x_a = g.loc[t, "ai"] if "ai" in g.columns and t in g.index else np.nan
    if pd.notna(x_h) and pd.notna(x_a):
        plt.plot([x_h, x_a], [i, i], color='0.6', linewidth=1)


human_color = sc_h.get_facecolor()[0]  # (r, g, b, a)

'''
human_color = "tab:blue"
ai_color = "tab:orange"
'''



# vertical dashed line for overall mean AI performance
mean_ai = fe_df_copy.loc[(fe_df_copy["leader_type"]=="ai"),"performance"].mean()
mean_ai_yes = fe_df_copy.loc[(fe_df_copy["leader_type"]=="ai") & (fe_df_copy["suggestions"]=="Yes"),"performance"].mean()
mean_ai_no = fe_df_copy.loc[(fe_df_copy["leader_type"]=="ai") & (fe_df_copy["suggestions"]=="No"),"performance"].mean()
mean_human = fe_df_copy.loc[fe_df_copy["leader_type"]=="human","performance"].mean()
mean_human_yes = fe_df_copy.loc[(fe_df_copy["leader_type"]=="human") & (fe_df_copy["suggestions"]=="Yes"),"performance"].mean()
mean_human_no = fe_df_copy.loc[(fe_df_copy["leader_type"]=="human") & (fe_df_copy["suggestions"]=="No"),"performance"].mean()
plt.axvline(x=mean_ai_performance, color="red", linestyle="--", linewidth=1.5, label="AI*")
plt.axvline(x=mean_human, color=human_color, linestyle="--", linewidth=1.5, label="H")
plt.axvline(x=mean_ai, color=ai_color, linestyle="--", linewidth=1.5, label="AI")
plt.axvline(x=mean_ai_yes, color=ai_color, linestyle="-.", linewidth=1.5, label="AI¹")
plt.axvline(x=mean_ai_no, color=ai_color, linestyle=":", linewidth=1.5, label="AI°")
plt.axvline(x=mean_human_yes, color=human_color, linestyle="-.", linewidth=1.5, label="H¹")
plt.axvline(x=mean_human_no, color=human_color, linestyle=":", linewidth=1.5, label="H°")

# labels and styling with larger fonts
plt.yticks(ypos, order.index, fontsize=12)  
plt.xlabel("Mean performance", fontsize=14, fontweight="bold")
plt.ylabel("Team #", fontsize=14, fontweight="bold")
plt.tick_params(axis="both", which="major", labelsize=12)
plt.legend(fontsize=12, title_fontsize=13)


plt.tight_layout()
plt.show()


means = fe_df.groupby(["team_id", "leader_type"])["performance"].mean().unstack()
missing_cols = [c for c in ["ai", "human"] if c not in means.columns]
if missing_cols:
    raise ValueError(f"leader_type must include both 'AI' and 'Human' (missing: {missing_cols})")

human_better_ids = means.index[means["human"] > means["ai"]].tolist()
ai_better_ids    = means.index[means["ai"] > means["human"]].tolist()



'''
res = test_ai_vs_human_per_layout(fe_df, performance_col="performance",
                                  leader_col="leader_type",
                                  layout_col="layout",
                                  team_col="team_id",      # set to None if you don't have it
                                  session_col=None,   # optional
                                  agg="median",
                                  min_n=4)
print(res)
'''




'''
# --- 2) pairwise_df ---
# Build from your tables (or read from CSV you exported). Must include columns listed in docstring.
# Example tiny mock (DELETE once you pass real data):


# Pick any pair label you have results for:
plot_top_deltas_for_pair(pairwise_df,
                         pair_label='0 vs 3',
                         k=12,
                         out_png='figure_top_deltas_0v3.png',
                         title_prefix="Largest |Δr| between sessions")
'''
'''
#Check diffence between when humans outperform AIs, do they improve or not
rng = np.random.default_rng(42)

# ---------- Helper effect sizes ----------
def cohens_d(x, y):
    x, y = np.asarray(x), np.asarray(y)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    # pooled SD (Welch’s variant)
    s = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)) if (nx+ny-2)>0 else np.nan
    return (x.mean() - y.mean()) / s if s and np.isfinite(s) else np.nan

def cliffs_delta(x, y):
    # δ = P(X>Y) - P(X<Y)
    x, y = np.asarray(x), np.asarray(y)
    n, m = len(x), len(y)
    if n == 0 or m == 0: return np.nan
    more = 0; less = 0
    for xi in x:
        more += np.sum(xi > y)
        less += np.sum(xi < y)
    return (more - less) / (n * m)

# ---------- 1) Identify Human-better vs AI-better teams ----------
means = fe_df.groupby(["team_id", "leader_type"])["performance"].mean().unstack()
missing_cols = [c for c in ["ai", "human"] if c not in means.columns]
if missing_cols:
    raise ValueError(f"leader_type must include both 'AI' and 'Human' (missing: {missing_cols})")

human_better_ids = means.index[means["human"] > means["ai"]].tolist()
ai_better_ids    = means.index[means["ai"] > means["human"]].tolist()
ties_ids         = means.index[means["human"] == means["ai"]].tolist()

print("Human-better teams:", human_better_ids)
print("AI-better teams   :", ai_better_ids)
if len(ties_ids):
    print("Ties (ignored here):", ties_ids)

# ---------- 2) Build per-team Human-led change (last - first) ----------
def human_delta_for_team(df_team):
    # df_team: rows for a single team, Human-led only
    if df_team.empty: return None
    # ensure ordered by session; get first and last Human-led sessions for that team
    df_team = df_team.sort_values("session")
    first = df_team.iloc[0]["performance"]
    last  = df_team.iloc[-1]["performance"]
    return float(last - first)

rows = []
for tid in fe_df["team_id"].unique():
    df_h = fe_df[(fe_df["team_id"] == tid) & (fe_df["leader_type"] == "human")]
    d = human_delta_for_team(df_h)
    if d is None or np.isnan(d): 
        continue
    group = "Human-better" if tid in human_better_ids else ("AI-better" if tid in ai_better_ids else "Tie")
    rows.append({"team_id": tid, "group": group, "delta_human": d})

delta_df = pd.DataFrame(rows)
delta_df = delta_df[delta_df["group"].isin(["Human-better","AI-better"])].copy()
delta_df["improved"] = (delta_df["delta_human"] > 0).astype(int)
print("\nPer-team Human-led Δ (last−first):\n", delta_df.sort_values(["group","team_id"]))

# ---------- 3) Group splits ----------
hb = delta_df[delta_df["group"]=="Human-better"]["delta_human"].to_numpy()
ab = delta_df[delta_df["group"]=="AI-better"]["delta_human"].to_numpy()
print(f"\nCounts — Human-better n={len(hb)}, AI-better n={len(ab)}")

# ---------- 4) Fisher’s exact on improvement proportions ----------
hb_imp = int((hb > 0).sum()); hb_not = len(hb) - hb_imp
ab_imp = int((ab > 0).sum()); ab_not = len(ab) - ab_imp
table = np.array([[hb_imp, hb_not],
                  [ab_imp, ab_not]])
oddsratio, p_fisher = fisher_exact(table, alternative="two-sided")
print("\nFisher’s exact (Improved vs Not):")
print("Contingency [[HB_improve, HB_not], [AB_improve, AB_not]] =", table.tolist())
print(f"Odds ratio = {oddsratio:.3f}, p = {p_fisher:.4f}")

# ---------- 5) Mean Δ comparison: Welch t-test + Mann–Whitney + effect sizes ----------
tt = ttest_ind(hb, ab, equal_var=False, alternative="greater")  # HB mean Δ > AB mean Δ ?
u  = mannwhitneyu(hb, ab, alternative="greater")                # nonparametric
d  = cohens_d(hb, ab)
cd = cliffs_delta(hb, ab)

print("\nMean Δ(Human-led) comparison (HB > AB?):")
print(f"Welch t-test: t = {tt.statistic:.3f}, p = {tt.pvalue:.4f}")
print(f"Mann–Whitney U: U = {u.statistic:.1f}, p = {u.pvalue:.4f}")
print(f"Effect sizes: Cohen’s d = {d:.3f}, Cliff’s δ = {cd:.3f}")
print(f"Group means: HB = {hb.mean():.3f} (n={len(hb)}), AB = {ab.mean():.3f} (n={len(ab)})")

# ---------- 6) Logistic regression: improved ~ group ----------
# Encode group: HB=1, AB=0
logit_df = delta_df.copy()
logit_df["HB"] = (logit_df["group"]=="Human-better").astype(int)
# Optional control: initial Human-led performance per team
init_h = (fe_df[fe_df["leader_type"]=="human"]
          .sort_values(["team_id","session"])
          .groupby("team_id").first()["performance"].rename("human_first"))
logit_df = logit_df.merge(init_h, left_on="team_id", right_index=True, how="left")

# Model A: improved ~ HB
mdlA = sm.Logit(logit_df["improved"], sm.add_constant(logit_df[["HB"]])).fit(disp=False)
print("\nLogistic regression A (improved ~ HB):")
print(mdlA.summary())
# Model B: improved ~ HB + human_first (control for starting point)
mdlB = sm.Logit(logit_df["improved"], sm.add_constant(logit_df[["HB","human_first"]])).fit(disp=False)
print("\nLogistic regression B (improved ~ HB + human_first):")
print(mdlB.summary())

# ---------- 7) Permutation tests ----------
B = 5000  # iterations
hb_mask = (delta_df["group"]=="Human-better").to_numpy()
deltas = delta_df["delta_human"].to_numpy()
impr   = delta_df["improved"].to_numpy()

obs_mean_gap = deltas[hb_mask].mean() - deltas[~hb_mask].mean()
obs_rate_gap = impr[hb_mask].mean()   - impr[~hb_mask].mean()

perm_mean_gaps = np.empty(B)
perm_rate_gaps = np.empty(B)

for b in range(B):
    perm_mask = rng.permutation(hb_mask)  # shuffle group labels
    perm_mean_gaps[b] = deltas[perm_mask].mean() - deltas[~perm_mask].mean()
    perm_rate_gaps[b] = impr[perm_mask].mean()   - impr[~perm_mask].mean()

p_perm_mean = (np.sum(perm_mean_gaps >= obs_mean_gap) + 1) / (B + 1)
p_perm_rate = (np.sum(perm_rate_gaps >= obs_rate_gap) + 1) / (B + 1)

print("\nPermutation tests (HB > AB):")
print(f"Mean Δ gap: observed = {obs_mean_gap:.3f}, p_perm = {p_perm_mean:.4f}")
print(f"Improvement-rate gap: observed = {obs_rate_gap:.3f}, p_perm = {p_perm_rate:.4f}")




# ---------- 2) Build per-team ai-led change (last - first) ----------
def ai_delta_for_team(df_team):
    if df_team.empty:
        return None
    df_team = df_team.sort_values("session")
    first = df_team.iloc[0]["performance"]
    last  = df_team.iloc[-1]["performance"]
    return float(last - first)

rows = []
for tid in fe_df["team_id"].unique():
    df_a = fe_df[(fe_df["team_id"] == tid) & (fe_df["leader_type"] == "ai")]
    d = ai_delta_for_team(df_a)
    if d is None or np.isnan(d):
        continue
    group = "ai_better" if tid in ai_better_ids else ("human_better" if tid in human_better_ids else "tie")
    rows.append({"team_id": tid, "group": group, "delta_ai": d})

delta_df = pd.DataFrame(rows)
delta_df = delta_df[delta_df["group"].isin(["ai_better","human_better"])].copy()
delta_df["improved"] = (delta_df["delta_ai"] > 0).astype(int)
print("\nai-led Δ (last−first) per team:\n", delta_df.sort_values(["group","team_id"]))

# ---------- 3) Group splits ----------
ab = delta_df[delta_df["group"]=="ai_better"]["delta_ai"].to_numpy()
hb = delta_df[delta_df["group"]=="human_better"]["delta_ai"].to_numpy()
print(f"\nCounts — ai_better n={len(ab)}, human_better n={len(hb)}")

# ---------- 4) Fisher’s exact on improvement proportions ----------
ab_imp = int((ab > 0).sum()); ab_not = len(ab) - ab_imp
hb_imp = int((hb > 0).sum()); hb_not = len(hb) - hb_imp
table = np.array([[ab_imp, ab_not],
                  [hb_imp, hb_not]])
oddsratio, p_fisher = fisher_exact(table, alternative="two-sided")
print("\nFisher’s exact (Improved vs Not) [rows: ai_better, human_better]:")
print("Contingency =", table.tolist())
print(f"Odds ratio = {oddsratio:.3f}, p = {p_fisher:.4f}")

# ---------- 5) Mean Δ comparison: Welch t-test + Mann–Whitney + effect sizes ----------
# Test if ai_better mean Δ > human_better mean Δ
tt = ttest_ind(ab, hb, equal_var=False, alternative="greater")
u  = mannwhitneyu(ab, hb, alternative="greater")
d  = cohens_d(ab, hb)
cd = cliffs_delta(ab, hb)

print("\nMean Δ(ai-led) comparison (ai_better > human_better?):")
print(f"Welch t-test: t = {tt.statistic:.3f}, p = {tt.pvalue:.4f}")
print(f"Mann–Whitney U: U = {u.statistic:.1f}, p = {u.pvalue:.4f}")
print(f"Effect sizes: Cohen’s d = {d:.3f}, Cliff’s δ = {cd:.3f}")
print(f"Group means: ai_better = {ab.mean():.3f} (n={len(ab)}), human_better = {hb.mean():.3f} (n={len(hb)})")

# ---------- 6) Logistic regression: improved ~ group (ai_better=1) ----------
logit_df = delta_df.copy()
logit_df["AB"] = (logit_df["group"]=="ai_better").astype(int)

# Optional control: initial ai-led performance per team
init_a = (fe_df[fe_df["leader_type"]=="ai"]
          .sort_values(["team_id","session"])
          .groupby("team_id").first()["performance"].rename("ai_first"))
logit_df = logit_df.merge(init_a, left_on="team_id", right_index=True, how="left")

# Model A: improved ~ AB
mdlA = sm.Logit(logit_df["improved"], sm.add_constant(logit_df[["AB"]])).fit(disp=False)
print("\nLogistic regression A (improved ~ AB):")
print(mdlA.summary())

# Model B: improved ~ AB + ai_first (control for starting point)
mdlB = sm.Logit(logit_df["improved"], sm.add_constant(logit_df[["AB","ai_first"]])).fit(disp=False)
print("\nLogistic regression B (improved ~ AB + ai_first):")
print(mdlB.summary())

# ---------- 7) Permutation tests (robustness) ----------
B = 5000
ab_mask = (delta_df["group"]=="ai_better").to_numpy()
deltas  = delta_df["delta_ai"].to_numpy()
impr    = delta_df["improved"].to_numpy()

obs_mean_gap = deltas[ab_mask].mean() - deltas[~ab_mask].mean()   # ai_better - human_better
obs_rate_gap = impr[ab_mask].mean()   - impr[~ab_mask].mean()

perm_mean_gaps = np.empty(B)
perm_rate_gaps = np.empty(B)

for b in range(B):
    perm_mask = rng.permutation(ab_mask)  # shuffle group labels
    perm_mean_gaps[b] = deltas[perm_mask].mean() - deltas[~perm_mask].mean()
    perm_rate_gaps[b] = impr[perm_mask].mean()   - impr[~perm_mask].mean()

p_perm_mean = (np.sum(perm_mean_gaps >= obs_mean_gap) + 1) / (B + 1)
p_perm_rate = (np.sum(perm_rate_gaps >= obs_rate_gap) + 1) / (B + 1)

print("\nPermutation tests (ai_better > human_better):")
print(f"Mean Δ gap: observed = {obs_mean_gap:.3f}, p_perm = {p_perm_mean:.4f}")
print(f"Improvement-rate gap: observed = {obs_rate_gap:.3f}, p_perm = {p_perm_rate:.4f}")

'''
# ---- Run it ----
#first_last_predictiveness(fe_df)

'''
means = fe_df.groupby(["team_id", "leader_type"])["performance"].mean().unstack()
missing_cols = [c for c in ["ai", "human"] if c not in means.columns]
human_better_ids = means.index[means["human"] > means["ai"]].tolist()
ai_better_ids    = means.index[means["ai"] > means["human"]].tolist()
only_better_humans = fe_df.loc[fe_df['team_id'].isin(human_better_ids)]
only_better_ais = fe_df.loc[fe_df['team_id'].isin(ai_better_ids)]
#compare_ai_vs_human_deltas(only_better_humans, target="performance", leader_col="leader_type", top_n=60)

res = compare_two_groups_human_deltas(
    df_a=only_better_humans,
    df_b=only_better_ais,
    target="performance",
    leader_col="leader_type",
    team_col="team_id",
    order_col="session",
    control_mode=None,             # or e.g. control_mode=["layout_complexity","avg_hazard"]
    exclude_contains=("relative_performance",),
    min_n=4,
    top_n=50
)
'''
'''
# 1) Compute per-layout r/p/n
corr_by_layout = correlations_per_layout_no_transitions_with_np(fe_df, layout_col='layout')

# 2) Write LaTeX with pairwise Fisher tests and FDR
write_per_layout_latex_with_meta_and_stats(
    corr_by_layout,
    out_tex="layout_session_corr_tables.tex",
    do_pairwise=True,
    strength_thresh=0.0,   # only test features with |r|>=0.20 in each layout
    min_n=4,
    fdr_scope="per_pair",   # or "global"
    top_k_pair=50
)
'''


compare_ai_vs_human(fe_df, target="performance", leader_col="leader_type", top_n=60)
#compare_ai_vs_human(fe_df, target="performance", leader_col="leader_type", top_n=60,control_mode="layout")
#compare_ai_vs_human(fe_df, target="performance", leader_col="leader_type", top_n=60, control_mode="complexity")

res = compute_corrs_by_leader(fe_df, target="performance", leader_col="leader_type", top_n=40)
#res = correlations_by_leader2(fe_df, target="performance", leader_col="leader_type", top_n=40, control_cols=["complexity"]) #,control_cols=["layout"])

#correlations_by_leader_and_suggestions(fe_df, target="performance", leader_col="leader_type", top_n=40, control_cols=["session"])
compare_suggestions_within_leader(fe_df)
'''
res = correlations_with_conditions(
    fe_df,
    target="performance",
    leader_col="leader_type",
    suggestions_col="suggestions",
    top_n=100
)
'''
res = compare_suggestions_within_each_leader_by_feature(fe_df, top_n=60)


#res = analyze_ai_advantage_print(fe_df, grain="team")
#plot_mean_differences_leader_type(fe_df)
#calculation_performance_human_vs_ai(fe_df)
#fe_df = fe_df[fe_df["team_id"] != 15]


###Calculate session performance
'''
# Function for Cliff's delta
def cliffs_delta(x, y):
    n, m = len(x), len(y)
    more = sum([1 for xi in x for yi in y if xi > yi])
    less = sum([1 for xi in x for yi in y if xi < yi])
    return (more - less) / (n * m)

# Function for Cohen's d (paired)
def cohens_d(x, y):
    diff = y - x
    return diff.mean() / diff.std(ddof=1)

results = []
sessions = sorted(fe_df["session"].unique())

for leader in fe_df["leader_type"].unique():
    df_sub = fe_df[fe_df["leader_type"] == leader]

    for i in range(len(sessions)-1):
        s1, s2 = sessions[i], sessions[i+1]

        perf1 = df_sub[df_sub.session == s1].sort_values("team_id")["performance"].reset_index(drop=True)
        perf2 = df_sub[df_sub.session == s2].sort_values("team_id")["performance"].reset_index(drop=True)

        min_len = min(len(perf1), len(perf2))
        perf1, perf2 = perf1[:min_len], perf2[:min_len]

        stat, p = wilcoxon(perf1, perf2)

        results.append({
            "leader_type": leader,
            "comparison": f"{s1}→{s2}",
            "mean_diff": (perf2 - perf1).mean(),
            "median_diff": (perf2 - perf1).median(),
            "wilcoxon_stat": stat,
            "p_value": p,
            "cliffs_delta": cliffs_delta(perf2, perf1),
            "cohens_d": cohens_d(perf1, perf2)
        })

res_df = pd.DataFrame(results)
print(res_df)
'''

pdb.set_trace()


res = per_team_performance_analysis_with_globals(
    fe_df,
    perf_col="performance",
    suggestions_col="suggestions",   # binary yes/no
    z_scope="group",                 # z-score within matching (leader, suggestions) group; use "global" if you prefer
    min_rows_corr=3,                 # global r needs at least 3 rows per feature
    do_fisher_within_team=True,      # Fisher only works when within-team n>=3 per leader (often not the case here)
    fdr_scope="team",
)

# 2) Inspect global “rules”
print_global_drivers(res, top_k=12)

# 3) See each team’s top drivers relative to those rules
print_team_vs_global(res, fe_df, top_k=10)

'''
survey_keys = [
    'Dependable', 'Reliable', 'Unresponsive', 'Predictable', 'Act consistently',
    'Function successfully', 'Malfunction', 'Have errors', 'Provide feedback',
    'Meet the needs of the mission/task', 'Provide appropriate information',
    'Communicate with people', 'Perform exactly as instructed', 'Follow directions',
    'Mental Demand: How mentally demanding was the task?',
    'Physical Demand: How physically demanding was the task?',
    'Temporal Demand: How hurried or rushed was the pace of the task?',
    'Performance: How successful were you in accomplishing what you were asked to do?',
    'Effort: How hard did you have to work to accomplish your level of performance?',
    'Frustration: How insecure, discouraged, irritated, stressed, and annoyed were you?'
]

results = analyze_survey_significance(
    fe_df,
    survey_keys=survey_keys,
    id_col="team_id",
    session_col="session",
    leader_col="leader_type",
    alpha=0.05,
    save_csv=True,        # set False if you don't want files
    outdir="."
)

friedman_df = results["friedman"]
kruskal_df  = results["kruskal"]
pairwise_df = results["pairwise"]

print(friedman_df)
print(kruskal_df)
print(pairwise_df)
'''
'''
# --- One-way ANOVA: does mean performance differ by team? ---
model = ols('performance ~ C(team_id)', data=fe_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# --- Post-hoc tests (Tukey HSD) to see which teams differ ---
comp = mc.MultiComparison(fe_df['performance'], fe_df['team_id'])
tukey_result = comp.tukeyhsd()
print(tukey_result)
'''
'''
#### demographic correlations (TODO: check with respect to leader not all teammates)
# Step 1: Aggregate performance by team
team_perf = fe_df.groupby("team_id")["performance"].agg(
    mean_perf="mean",
    std_perf="std",
    first_perf=lambda x: x.iloc[0],
    last_perf=lambda x: x.iloc[-1]
).reset_index()
team_perf["delta_perf"] = team_perf["last_perf"] - team_perf["first_perf"]

survey_fields = demographic_qs

meta = fe_df[["team_id"] + survey_fields].drop_duplicates("team_id")
merged = team_perf.merge(meta, on="team_id", how="left")
# Step 3: Correlation with numeric/ordinal survey responses


outcomes = ["mean_perf", "std_perf", "delta_perf"]
results = []

for outcome in outcomes:
    for field in survey_fields:
        rho, pval = spearmanr(merged[field], merged[outcome], nan_policy="omit")
        results.append({
            "outcome": outcome,
            "field": field,
            "rho": rho,
            "pval": pval
        })

corrs_df = pd.DataFrame(results)
print(corrs_df.sort_values(["outcome", "rho"], ascending=[True, False]))
'''

ID_COLS_DEFAULT = {
    "team_id", "session", "leader_type", "layout", "suggestions", "name",
    "performance", "delta_perf", "mean_perf", "std_perf",
}

def ensure_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    need_delta = "delta_perf" not in df.columns
    need_mean  = "mean_perf"  not in df.columns
    need_std   = "std_perf"   not in df.columns
    if not (need_delta or need_mean or need_std):
        return df

    req = {"team_id","session","performance"}
    if not req.issubset(df.columns):
        raise ValueError(
            "CSV lacks precomputed outcomes and also lacks columns "
            "'team_id','session','performance' needed to compute them."
        )

    # make sure 'performance' is numeric
    df = df.copy()
    df["performance"] = pd.to_numeric(df["performance"], errors="coerce")

    g = df.groupby("team_id", as_index=False)

    # mean/std per team
    if need_mean or need_std:
        agg = g["performance"].agg(mean_perf="mean", std_perf="std")
        df = df.merge(agg, on="team_id", how="left")

    # robust delta (last - first by session), no .apply/.rename ambiguity
    if need_delta:
        # sort once, then take first/last
        tmp = (df.sort_values(["team_id","session"])
                 .groupby("team_id")["performance"]
                 .agg(first="first", last="last")
                 .assign(delta_perf=lambda d: d["last"] - d["first"])
                 [["delta_perf"]]
                 .reset_index())
        df = df.merge(tmp, on="team_id", how="left")

    return df

def pick_predictors(df: pd.DataFrame, exclude: set):
    preds = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c]
            if s.notna().sum() >= 3 and s.nunique(dropna=True) >= 2:
                preds.append(c)
    return preds

def corr_all(df: pd.DataFrame, outcomes, predictors):
    # reduce to one row per team if multiple rows exist (so factors aren’t duplicated)
    key = "team_id" if "team_id" in df.columns else None
    if key:
        # average numeric predictors per team; outcomes kept from first (they’re constant per team after ensure_outcomes)
        num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
        base = (df.groupby(key)[outcomes].first().reset_index()
                .merge(df.groupby(key)[num_cols].mean().reset_index(), on=key, how="left"))
    else:
        base = df.copy()

    rows = []
    for y_col in outcomes:
        y = base[y_col].to_numpy(dtype=float)
        for x_col in predictors:
            x = base[x_col].to_numpy(dtype=float)
            mask = ~np.isnan(x) & ~np.isnan(y)
            if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
                continue
            rho, p = spearmanr(x[mask], y[mask])
            rows.append({"outcome": y_col, "field": x_col, "rho": float(rho), "pval": float(p)})
    res = pd.DataFrame(rows)
    if not res.empty:
        res["abs_rho"] = res["rho"].abs()
        res = res.sort_values(["outcome","abs_rho"], ascending=[True, False]).reset_index(drop=True)
    return res

def topk_tables(res: pd.DataFrame, k=10, abs_thresh=0.0):
    tops_pos, tops_neg = [], []
    for y in res["outcome"].unique():
        sub = res[res["outcome"]==y].copy()
        if abs_thresh > 0:
            sub = sub[sub["abs_rho"] >= abs_thresh]
        pos = sub.sort_values(["rho","pval"], ascending=[False, True]).head(k).copy()
        pos.insert(0, "direction", "+")
        neg = sub.sort_values(["rho","pval"], ascending=[True, True]).head(k).copy()
        neg.insert(0, "direction", "-")
        tops_pos.append(pos); tops_neg.append(neg)
    return (pd.concat(tops_pos, ignore_index=True) if tops_pos else pd.DataFrame(),
            pd.concat(tops_neg,  ignore_index=True) if tops_neg  else pd.DataFrame())

def write_latex(res, top_pos, top_neg, out_tex):
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated: Spearman correlations between factors and outcomes\n\n")
        for y in res["outcome"].unique():
            sub = res[res["outcome"]==y][["field","rho","pval","abs_rho"]].copy()
            sub["rho"] = sub["rho"].round(3)
            sub["pval"] = sub["pval"].apply(lambda v: f"{v:.3f}")
            sub["abs_rho"] = sub["abs_rho"].round(3)
            f.write(f"%% ---- Outcome: {y} ----\n")
            f.write(sub.to_latex(index=False, caption=f"Spearman correlations for {y}",
                                 label=f"tab:corr_all_{y}"))
            f.write("\n\n")
        for df, name in [(top_pos,"Top positive correlations by outcome"),
                         (top_neg,"Top negative correlations by outcome")]:
            if df.empty: continue
            d2 = df.copy()
            d2["rho"] = d2["rho"].round(3)
            d2["pval"] = d2["pval"].apply(lambda v: f"{float(v):.3f}")
            d2["abs_rho"] = d2["abs_rho"].round(3)
            f.write(f"%% ---- {name} ----\n")
            f.write(d2[["outcome","direction","field","rho","pval","abs_rho"]]
                    .to_latex(index=False, caption=name, label=f"tab:{name.lower().replace(' ','_')}"))
            f.write("\n\n")


def write_samelead_corr_tables_with_fisher(
    res_same_prev: Dict[str, any],
    out_tex: str = "samelead_corr_tables.tex",
    thresh: float = 0.40,                 # |r|>thresh for the first two tables
    top_diff_k: int = 10,                 # how many rows to show in the “most different” table
    exclude: Optional[List[str]] = None,  # e.g., ["Δ_dangerous_weighted_distance"]
    fdr_scope: str = "global",            # "global" (over all compared features) or "none"
    min_n: int = 4                        # minimum n for Fisher test per group
) -> Dict[str, pd.DataFrame]:
    """
    Writes LaTeX tables comparing HUMAN vs AI leadership correlations with Δperformance, and
    returns the DataFrames used.

    Inputs:
      res_same_prev: dict produced by raw_delta_descriptives_flexible(..., same_leadership_only=True, ...)
                     Must contain 'delta_df' (for recomputing r/p/n) OR
                     'corr_by_leader': {'human': df, 'ai': df} with at least ['feature','r' or 'corr_with_Δperf'].
      out_tex: output LaTeX filename
      thresh: absolute r threshold for HUMAN/AI individual tables
      top_diff_k: rows to display in the “most different correlations” table
      exclude: list of feature names to remove (e.g., tautological features)
      fdr_scope: "global" to apply BH-FDR across all tested features; "none" to skip FDR
      min_n: minimum sample size per group for Fisher test

    Returns:
      {
        'human_corr': DataFrame[feature, r, p, n],
        'ai_corr':    DataFrame[feature, r, p, n],
        'diff_full':  DataFrame with Δρ, z_diff, p_diff, q_diff for all comparable features,
        'diff_top':   Display-trimmed/rounded top-diff DataFrame written to LaTeX
      }
    """
    exclude = set(exclude or [])

    # ----------------- helpers -----------------
    def _corrs_rpn_from_delta(delta_df: pd.DataFrame, leader: str) -> pd.DataFrame:
        sub = delta_df[(delta_df['leader_type'] == leader) & delta_df['Δ_performance'].notna()].copy()
        dcols = [c for c in sub.columns if c.startswith('Δ_') and c != 'Δ_performance']
        rows = []
        for feat in dcols:
            xy = sub[[feat, 'Δ_performance']].dropna()
            if len(xy) >= 3 and xy[feat].nunique() > 1 and xy['Δ_performance'].nunique() > 1:
                r, p = spearmanr(xy[feat], xy['Δ_performance'])
                rows.append({'feature': feat, 'r': float(r), 'p': float(p), 'n': int(len(xy))})
        out = pd.DataFrame(rows)
        if out.empty:
            return pd.DataFrame(columns=['feature','r','p','n'])
        return out.sort_values('r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    def _normalize_corr_df(df: pd.DataFrame) -> pd.DataFrame:
        """Accepts either columns ['feature','r','p','n'] or ['feature','corr_with_Δperf']; adds p,n if missing."""
        out = df.copy()
        if 'r' not in out.columns:
            if 'corr_with_Δperf' in out.columns:
                out = out.rename(columns={'corr_with_Δperf': 'r'})
            else:
                raise ValueError("Correlation frame must contain either 'r' or 'corr_with_Δperf'.")
        if 'p' not in out.columns:
            out['p'] = np.nan
        if 'n' not in out.columns:
            out['n'] = np.nan
        out = out[['feature','r','p','n']]
        out['r'] = pd.to_numeric(out['r'], errors='coerce')
        out['p'] = pd.to_numeric(out['p'], errors='coerce')
        out['n'] = pd.to_numeric(out['n'], errors='coerce')
        out = out.dropna(subset=['feature','r'])
        return out

    def _prep_corr_df_rpn(df: pd.DataFrame) -> pd.DataFrame:
        out = _normalize_corr_df(df)
        if exclude:
            out = out[~out['feature'].isin(exclude)]
        out['abs_corr'] = out['r'].abs()
        return out

    def _fisher_z(r: float) -> float:
        r = float(np.clip(r, -0.999999, 0.999999))
        return 0.5 * np.log((1 + r) / (1 - r))

    def _fisher_pdiff(rA: float, nA: float, rB: float, nB: float) -> Tuple[float, float]:
        if pd.isna(rA) or pd.isna(rB) or pd.isna(nA) or pd.isna(nB):
            return (np.nan, np.nan)
        nA, nB = int(nA), int(nB)
        if nA < min_n or nB < min_n:
            return (np.nan, np.nan)
        zA, zB = _fisher_z(rA), _fisher_z(rB)
        se = np.sqrt(1/(nA - 3) + 1/(nB - 3))
        z = (zA - zB) / se
        p = 2 * (1 - norm.cdf(abs(z)))
        return (z, p)

    # ----------------- get corrs for each leadership -----------------
    # Prefer recompute from delta_df to get n & p; else use provided frames.
    if 'delta_df' in res_same_prev and isinstance(res_same_prev['delta_df'], pd.DataFrame):
        delta_df = res_same_prev['delta_df'].copy()
        human_df = _corrs_rpn_from_delta(delta_df, 'human')
        ai_df    = _corrs_rpn_from_delta(delta_df, 'ai')
    else:
        # Fall back to precomputed
        human_df = res_same_prev['corr_by_leader']['human'].copy()
        ai_df    = res_same_prev['corr_by_leader']['ai'].copy()
        human_df = _normalize_corr_df(human_df)
        ai_df    = _normalize_corr_df(ai_df)

    human_prepped = _prep_corr_df_rpn(human_df)
    ai_prepped    = _prep_corr_df_rpn(ai_df)

    # ----------------- tables 1 & 2: |r| > thresh -----------------
    human_filtered = human_prepped #[human_prepped['abs_corr'] > thresh][['feature','r']].copy()
    ai_filtered    = ai_prepped #[ai_prepped['abs_corr'] > thresh][['feature','r']].copy()
    human_filtered = human_filtered.rename(columns={'r':'corr_with_Δperf'})
    ai_filtered    = ai_filtered.rename(columns={'r':'corr_with_Δperf'})

    # ----------------- table 3: MOST DIFFERENT with Fisher p_diff & q_diff -----------------
    diff_tbl = (human_prepped[['feature','r','p','n']].rename(columns={'r':'human_r','p':'p_A','n':'n_A'})
                .merge(ai_prepped[['feature','r','p','n']].rename(columns={'r':'ai_r','p':'p_B','n':'n_B'}),
                       on='feature', how='inner'))

    diff_tbl['Δcorr_ai_minus_human'] = diff_tbl['ai_r'] - diff_tbl['human_r']
    diff_tbl['|Δcorr|'] = diff_tbl['Δcorr_ai_minus_human'].abs()

    # --- NEW: filter here ---
    diff_tbl = diff_tbl[
        (diff_tbl['|Δcorr|'] > thresh) &
        (
            (diff_tbl['human_r'].abs() >= thresh) |
            (diff_tbl['ai_r'].abs() >= thresh)
        )
    ].copy()

    # Fisher per feature
    z_list, p_list = [], []
    for row in diff_tbl.itertuples(index=False):
        z, p = _fisher_pdiff(row.human_r, row.n_A, row.ai_r, row.n_B)
        z_list.append(z); p_list.append(p)
    diff_tbl['z_diff'] = z_list
    diff_tbl['p_diff'] = p_list

    # FDR
    if fdr_scope == "global":
        mask = diff_tbl['p_diff'].notna()
        diff_tbl['q_diff'] = np.nan
        if mask.any():
            _, q, _, _ = multipletests(diff_tbl.loc[mask, 'p_diff'], method='fdr_bh')
            diff_tbl.loc[mask, 'q_diff'] = q
    else:
        diff_tbl['q_diff'] = np.nan  # no FDR

    # Prepare pretty outputs
    human_out = human_filtered.copy()
    human_out['corr_with_Δperf'] = human_out['corr_with_Δperf'].round(3)

    ai_out = ai_filtered.copy()
    ai_out['corr_with_Δperf'] = ai_out['corr_with_Δperf'].round(3)

    diff_out = (diff_tbl
                .sort_values('|Δcorr|', ascending=False)
                .head(top_diff_k)
                .copy())

    for c in ['human_r','ai_r','Δcorr_ai_minus_human','|Δcorr|','z_diff','p_diff','q_diff']:
        if c in diff_out.columns:
            diff_out[c] = diff_out[c].astype(float).round(3)

    # ----------------- write LaTeX -----------------
    with open(out_tex, "w") as f:
        f.write("%% Auto-generated tables: correlations by leadership (same-leadership contrast analysis)\n\n")

        f.write("%% -------- HUMAN-LED (|r| > %.2f) --------\n" % thresh)
        f.write(human_out.to_latex(index=False, float_format="%.3f",
                                   caption="Human-led: Spearman correlations with $\\Delta$performance (|r| > %.2f)" % thresh,
                                   label="tab:humn_corr_gt_%.0f" % (thresh*100)))
        f.write("\n\n")

        f.write("%% -------- AI-LED (|r| > %.2f) --------\n" % thresh)
        f.write(ai_out.to_latex(index=False, float_format="%.3f",
                                caption="AI-led: Spearman correlations with $\\Delta$performance (|r| > %.2f)" % thresh,
                                label="tab:ai_corr_gt_%.0f" % (thresh*100)))
        f.write("\n\n")

        f.write("%% -------- MOST DIFFERENT BETWEEN HUMAN vs AI (with Fisher p_diff & q_diff) --------\n")
        cols = ['feature','human_r','ai_r','Δcorr_ai_minus_human','|Δcorr|','z_diff','p_diff','q_diff']
        f.write(diff_out[cols].to_latex(index=False, float_format="%.3f",
                                        caption="Largest differences in correlations between AI-led and Human-led transitions (with Fisher test).",
                                        label="tab:largest_diff_corr"))
        f.write("\n")

    print(f"LaTeX tables written to: {out_tex}")

    # Return raw & pretty tables for further use
    return {
        'human_corr': human_prepped[['feature','r','p','n']].sort_values('r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True),
        'ai_corr':    ai_prepped[['feature','r','p','n']].sort_values('r', key=lambda s: s.abs(), ascending=False).reset_index(drop=True),
        'diff_full':  diff_tbl.sort_values('|Δcorr|', ascending=False).reset_index(drop=True),
        'diff_top':   diff_out.reset_index(drop=True)
    }

def per_team_driver_report_single_row(
    delta_df: pd.DataFrame,               # one row per (team, leader) with Δ_* columns
    leader_col: str = "leader_type",
    team_col: str = "team_id",
    target_col: str = "Δ_performance",
    global_corrs: dict = None,            # {'human': DataFrame[feature,r], 'ai': DataFrame[feature,r]}
    top_k: int = 8
):
    """
    Attribution-style report when each team has only ONE delta row.
    Uses global Spearman correlations r as signed importances, and z-scores per feature
    (across all teams) to compute per-team driver scores.

    score = z_feature * r * |r|   # emphasizes features both extreme (z) and globally important (|r|)
    Returns: dict[(team, leader)] -> DataFrame[feature, z, r, score] with top positive/negative drivers.
    """

    if global_corrs is None or not {'human','ai'}.issubset(global_corrs.keys()):
        raise ValueError("Provide global_corrs={'human':df, 'ai':df} with columns ['feature','r'].")

    # collect Δ_* columns and z-score across teams (ignoring target)
    delta_cols = [c for c in delta_df.columns if c.startswith("Δ_") and c != target_col]
    stats = delta_df[delta_cols].agg(['mean','std'])
    means = stats.loc['mean']; stds = stats.loc['std'].replace(0, np.nan)

    # build lookup for r by leader
    r_map = {}
    for lt in ['human','ai']:
        df = global_corrs[lt].copy()
        if 'r' not in df.columns:
            if 'corr_with_Δperf' in df.columns:
                df = df.rename(columns={'corr_with_Δperf':'r'})
            else:
                raise ValueError("global_corrs frames must have 'r' or 'corr_with_Δperf'.")
        r_map[lt] = dict(zip(df['feature'], df['r']))

    reports = {}
    for idx, row in delta_df.iterrows():
        lt = str(row[leader_col]).lower()
        tid = row[team_col]
        if lt not in r_map:
            continue

        # compute z-scores for this row
        vals = row[delta_cols].astype(float)
        z = (vals - means) / stds
        z = z.replace([np.inf, -np.inf], np.nan)

        # align r’s (missing r -> 0)
        r_series = pd.Series({f: r_map[lt].get(f, 0.0) for f in delta_cols})
        # contribution score
        score = z * r_series * r_series.abs()

        # assemble table
        tbl = pd.DataFrame({
            'feature': delta_cols,
            'z': z.values,
            'r': r_series.values,
            'score': score.values
        }).dropna(subset=['z'])

        # rank: top positive drivers and top negative “draggers”
        tbl = tbl.sort_values('score', ascending=False)
        top_pos = tbl.head(top_k)
        top_neg = tbl.sort_values('score').head(top_k)

        # combine for a concise report
        rep = pd.concat([
            top_pos.assign(kind='driver(+)'),
            top_neg.assign(kind='driver(-)')
        ]).sort_values(['kind','score'], ascending=[True, False]).reset_index(drop=True)

        reports[(tid, lt)] = rep

    return reports



df = ensure_outcomes(fe_df)

outcomes = [c for c in ["delta_perf","mean_perf","std_perf"] if c in df.columns]
if not outcomes:
    raise ValueError("No outcomes found/constructed (delta_perf, mean_perf, std_perf).")

exclude = set(ID_COLS_DEFAULT)
predictors = [c for c in df.columns
              if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
# drop outcome leakage if someone stored them twice
predictors = [c for c in predictors if c not in outcomes]

out_prefix = "out"

res = corr_all(df, outcomes, predictors)
res.to_csv(f"{out_prefix}_full.csv", index=False)

topk = 20
top_pos, top_neg = topk_tables(res, k=topk, abs_thresh=0.4)
top_pos.to_csv(f"{out_prefix}_top_pos.csv", index=False)
top_neg.to_csv(f"{out_prefix}_top_neg.csv", index=False)

write_latex(res, top_pos, top_neg, f"{out_prefix}.tex")

print(f"Wrote {out_prefix}_full.csv, {out_prefix}_top_pos.csv, {out_prefix}_top_neg.csv")
for y in outcomes:
    sub = res[res["outcome"]==y]
    if sub.empty: continue
    print(f"\nOutcome: {y}")
    print(" Top +:\n", sub.sort_values("rho", ascending=False).head(topk)[["field","rho","pval"]])
    print(" Top -:\n", sub.sort_values("rho", ascending=True ).head(topk)[["field","rho","pval"]])




# 1) Non-contiguous, previous-with-same-leader (recommended)
res_same_prev = raw_delta_descriptives_flexible(
    fe_df,
    pair_mode="prev",
    same_leadership_only=True,   # compare only within same leader type
    keep_flats=False             # drop ~0 changes for cleaner contrasts
)
res_same_prev['delta_df'].to_csv("team_flexible_transitions_samelead.csv", index=False)

res = write_samelead_corr_tables_with_fisher(
    res_same_prev,
    out_tex="samelead_corr_tables.tex",
    thresh=0.4,
    top_diff_k=50,
    exclude=["Δ_dangerous_weighted_distance"],
    fdr_scope="global",
    min_n=4
)

# global correlations from your earlier step (Spearman per leadership)
human_corrs = res_same_prev['corr_by_leader']['human'][['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'r'})
ai_corrs    = res_same_prev['corr_by_leader']['ai'][['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'r'})

global_corrs = {'human': human_corrs, 'ai': ai_corrs}

# delta_df should have one row per team (and leader), with Δ_* columns and Δ_performance
reports = per_team_driver_report_single_row(res_same_prev['delta_df'], global_corrs=global_corrs, top_k=6)

'''
# Example: print a team’s drivers
for (tid, lt), rep in reports.items():
    print(f"\n=== Team {tid} | Leader: {lt} ===")
    print(rep[['kind','feature','z','r','score']].round(3).to_string(index=False))
'''

# 2) previous
res_fixed_any = raw_delta_descriptives_flexible(
    fe_df,
    pair_mode="fixed",
    same_leadership_only=False,   # compare only within same leader type
    keep_flats=False             # drop ~0 changes for cleaner contrasts
)



write_fixed_transition_latex_with_meta(
    res_fixed_any,
    out_tex="fixed_transition_corr_tables.tex",
    note_excluded="fixed pairs; excluding dangerous_weighted_distance",
    top_k_meta=50,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.4,
    top_k_pair=50,
    top_k_contrast=50,
    strength_thresh=0.0
)

write_session_corr_latex_with_meta(
    res_any=res_fixed_any,
    out_tex="session_corr_tables.tex",
    note_excluded="per-session correlations; excludes ids & Δ_*",
    exclude_features=[
        # optional: drop tautologies/noisy stuff here
        # "dangerous_weighted_distance", "dangerous_weighted_work",
    ],
    abs_thresh_single=0.0,   # only show |r|>=.30 in per-session tables & meta
    top_k_meta=50,
    abs_thresh_pair=0.0,     # keep |r|>=.30 in BOTH sessions before testing
    strength_thresh=0.0,     # and require |r|>=.40 in BOTH for pair outputs
    diff_thresh_pair=0.5,    # call “dissimilar” if same-sign but |Δr|>=.40
    top_k_pair=50
)

'''
res_same_prev['delta_df'].to_csv("team_flexible_transitions_samelead.csv", index=False)
res_fixed_any['delta_df'].to_csv("team_fixed_transitions_anylead.csv", index=False)

with open("samelead_report.txt", "w") as f:
    f.write("=== HUMAN up vs down (same-leadership, non-contiguous) ===\n")
    f.write(res_same_prev['updown_by_leader']['human'].to_string())
    f.write("\n\n=== AI up vs down (same-leadership, non-contiguous) ===\n")
    f.write(res_same_prev['updown_by_leader']['ai'].to_string())

    f.write("\n\n=== HUMAN improved − declined contrast ===\n")
    f.write(res_same_prev['contrast_by_leader']['human'].to_string())
    f.write("\n\n=== AI improved − declined contrast ===\n")
    f.write(res_same_prev['contrast_by_leader']['ai'].to_string())

    f.write("\n\n=== HUMAN correlations with Δperformance ===\n")
    f.write(res_same_prev['corr_by_leader']['human'].to_string())
    f.write("\n\n=== AI correlations with Δperformance ===\n")
    f.write(res_same_prev['corr_by_leader']['ai'].to_string())

    f.write("\n\n=== Raw delta rows (head) ===\n")
    f.write(res_same_prev['delta_df'].head().to_string())

#LATEX STARTS
# --- inputs from your pipeline ---
# res_same_prev = raw_delta_descriptives_flexible(..., same_leadership_only=True, ...)
human_df = res_same_prev['corr_by_leader']['human'].copy()
ai_df    = res_same_prev['corr_by_leader']['ai'].copy()

# Optional: exclude tautological or noisy items (uncomment if desired)
EXCLUDE = set([
    "Δ_dangerous_weighted_distance",   # basically the metric itself
    # add more to exclude if you want
])

THRESH = 0.40        # abs correlation threshold for the first 2 tables
TOP_DIFF_K = 10      # how many most-different features to show in the third table
OUT_TEX = "samelead_corr_tables.tex"

# --- small helpers ---
def _prep_corr_df(df):
    out = df.copy()
    out = out[~out['feature'].isin(EXCLUDE)]
    # ensure numeric
    out['corr_with_Δperf'] = pd.to_numeric(out['corr_with_Δperf'], errors='coerce')
    # sort by absolute correlation (desc)
    out = out.assign(abs_corr=lambda d: d['corr_with_Δperf'].abs()) \
             .sort_values('abs_corr', ascending=False)
    return out

human_prepped = _prep_corr_df(human_df)
ai_prepped    = _prep_corr_df(ai_df)

# Filter by |r| > THRESH for the first two tables
human_filtered = human_prepped[human_prepped['abs_corr'] > THRESH].drop(columns=['abs_corr'])
ai_filtered    = ai_prepped[ai_prepped['abs_corr'] > THRESH].drop(columns=['abs_corr'])

# --- Build the "most different correlations" table ---
# Merge on feature (inner join so we compare same feature across leaderships)
diff_tbl = (human_prepped[['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'human_corr'})
            .merge(ai_prepped[['feature','corr_with_Δperf']].rename(columns={'corr_with_Δperf':'ai_corr'}),
                   on='feature', how='inner'))

# Compute differences and sort by absolute difference
diff_tbl['Δcorr_ai_minus_human'] = diff_tbl['ai_corr'] - diff_tbl['human_corr']
diff_tbl['|Δcorr|'] = diff_tbl['Δcorr_ai_minus_human'].abs()
diff_tbl = diff_tbl.sort_values('|Δcorr|', ascending=False).head(TOP_DIFF_K)

# Optional: nice rounding for display (we'll keep full precision in data)
human_out = human_filtered.copy()
human_out['corr_with_Δperf'] = human_out['corr_with_Δperf'].round(3)

ai_out = ai_filtered.copy()
ai_out['corr_with_Δperf'] = ai_out['corr_with_Δperf'].round(3)

diff_out = diff_tbl.copy()
for c in ['human_corr','ai_corr','Δcorr_ai_minus_human','|Δcorr|']:
    diff_out[c] = diff_out[c].round(3)

# --- Write LaTeX file ---
with open(OUT_TEX, "w") as f:
    f.write("%% Auto-generated tables: correlations by leadership (same-leadership contrast analysis)\n\n")

    f.write("%% -------- HUMAN-LED (|r| > %.2f) --------\n" % THRESH)
    f.write(human_out.to_latex(index=False, float_format="%.3f",
                               caption="Human-led: correlations with $\\Delta$performance (|r| > %.2f)" % THRESH,
                               label="tab:humn_corr_gt_%.0f" % (THRESH*100)))
    f.write("\n\n")

    f.write("%% -------- AI-LED (|r| > %.2f) --------\n" % THRESH)
    f.write(ai_out.to_latex(index=False, float_format="%.3f",
                            caption="AI-led: correlations with $\\Delta$performance (|r| > %.2f)" % THRESH,
                            label="tab:ai_corr_gt_%.0f" % (THRESH*100)))
    f.write("\n\n")

    f.write("%% -------- MOST DIFFERENT BETWEEN HUMAN vs AI --------\n")
    # Reorder columns for readability
    cols = ['feature', 'human_corr', 'ai_corr', 'Δcorr_ai_minus_human', '|Δcorr|']
    f.write(diff_out[cols].to_latex(index=False, float_format="%.3f",
                                    caption="Largest differences in correlations between AI-led and Human-led transitions",
                                    label="tab:largest_diff_corr"))
                                    
                                   
    f.write("\n\n")
    
    f.write("%% -------- MOST DIFFERENT (absolute correlation values) --------\n")
    diff_tbl_abs = diff_tbl.copy()
    diff_tbl_abs['Δ|corr|'] = diff_tbl_abs['ai_corr'].abs() - diff_tbl_abs['human_corr'].abs()
    diff_tbl_abs['|Δ|corr||'] = diff_tbl_abs['Δ|corr|'].abs()
    diff_tbl_abs = diff_tbl_abs.sort_values('|Δ|corr||', ascending=False).head(TOP_DIFF_K)

    cols_abs = ['feature', 'human_corr', 'ai_corr', 'Δ|corr|', '|Δ|corr||']
    f.write(diff_tbl_abs[cols_abs].to_latex(index=False, float_format="%.3f",
                                            caption="Largest differences in *absolute* correlations between AI-led and Human-led transitions",
                                            label="tab:largest_absdiff_corr"))
    f.write("\n")
    

TOP_K = 10


# Pull your existing contrast frames
human_contrast = res_same_prev['contrast_by_leader']['human'].copy()
ai_contrast    = res_same_prev['contrast_by_leader']['ai'].copy()

# Build top-k tables
h_col, human_top_impr, human_top_decl = _top_k_pos_neg(human_contrast, TOP_K)
a_col, ai_top_impr,    ai_top_decl    = _top_k_pos_neg(ai_contrast, TOP_K)

with open(OUT_TEX, "a") as f:  # append to your samelead_corr_tables.tex
    f.write("\n%% -------- HUMAN: Top-k improved vs declined (same-leadership) --------\n")
    f.write(human_top_impr.to_latex(
        index=False, float_format="%.3f",
        caption=f"Human-led: top {TOP_K} factors when performance improved (highest positive {h_col}).",
        label="tab:human_topk_improved"
    ))
    f.write("\n\n")
    f.write(human_top_decl.to_latex(
        index=False, float_format="%.3f",
        caption=f"Human-led: top {TOP_K} factors when performance declined (most negative {h_col}).",
        label="tab:human_topk_declined"
    ))
    f.write("\n\n")

    f.write("%% -------- AI: Top-k improved vs declined (same-leadership) --------\n")
    f.write(ai_top_impr.to_latex(
        index=False, float_format="%.3f",
        caption=f"AI-led: top {TOP_K} factors when performance improved (highest positive {a_col}).",
        label="tab:ai_topk_improved"
    ))
    f.write("\n\n")
    f.write(ai_top_decl.to_latex(
        index=False, float_format="%.3f",
        caption=f"AI-led: top {TOP_K} factors when performance declined (most negative {a_col}).",
        label="tab:ai_topk_declined"
    ))
    f.write("\n")
print(f"LaTeX tables written to: {OUT_TEX}")

#LATEX ENDs
'''

# df_sessions already built in your script (list of dicts -> DataFrame)
res = correlations_per_layout_no_transitions(
    df_sessions=fe_df,
    layout_col='layout',            # or 'map' if that’s your column
)
write_per_layout_latex(res['corr_by_layout'], out_tex="layout_session_corr_tables.tex")

write_per_layout_latex_with_meta(res['corr_by_layout'], out_tex="layout_session_corr_tables.tex")

# optional heatmap
plot_layout_heatmap(res['corr_by_layout'], top_k=10, out_png="layout_corr_heatmap.png")


corrs = correlations_by_leadership_and_suggestions(fe_df)

for (lt, sug), tbl in corrs.items():
    print(f"\nLeader={lt}, Suggestions={sug}")
    print(tbl.head(10))
    
write_by_leadership_suggestions_latex_with_meta(
    corrs,        # dict like {('human','on'): df, ('ai','off'): df, ...}
    out_tex="lead_sugg_corr_tables.tex",
    note_excluded="excluding dangerous_weighted_distance",
    top_k=50,
    require_all_groups=True,
    pretty_names={
        ('human','on'):  "Human, Suggestions ON",
        ('human','off'): "Human, Suggestions OFF",
        ('ai','on'):     "AI, Suggestions ON",
        ('ai','off'):    "AI, Suggestions OFF",
    },
    do_pairwise=True,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.40,
    top_k_pair=50,
    raw_df=fe_df
)
results_delta = correlations_by_transition(fe_df, use_delta=True)  # correlations to Δperformance
for trans, corr_df in results_delta.items():
    print(f"\n=== {trans} — correlations w.r.t. Δperformance ===")
    print(corr_df.head(20))
    corr_df.to_csv(f"corr_{trans.replace('->','to')}_delta_performance.csv", index=False)

save_latex_tables(results_delta, filename_prefix="corr_delta", caption_suffix="$\Delta$performance")

write_by_transition_latex_with_meta(
    results_delta,
    out_tex="transition_corr_tables_raw.tex",
    note_excluded="excluding dangerous_weighted_distance",
    top_k=50,
    do_pairwise=True,
    abs_thresh_pair=0.0,
    diff_thresh_pair=0.40,
    top_k_pair=50,
    # (optional prettier names)
    pretty_names={
        "AI->Human":   "AI → Human",
        "Human->AI":   "Human → AI",
        "AI->AI":      "AI → AI",
        "Human->Human":"Human → Human",
    }
)

