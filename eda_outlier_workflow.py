"""
End-to-End Outlier Detection and Distribution Profiling Workflow
Amazon ML Challenge 2026: Business Entity Resolution
Lead ML Engineer & Principal Data Scientist Analysis Script
"""

import os
import sys
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import mahalanobis
from sklearn.covariance import EmpiricalCovariance, EllipticEnvelope, MinCovDet
import matplotlib.pyplot as plt
import seaborn as sns

# Set strict global random state and visual styling
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15
})

# Output directories
OUTPUT_DIR = "eda_artifacts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------
# Step 1: Raw Data Ingestion & Continuous Feature Extraction
# ---------------------------------------------------------
print("="*70)
print("STEP 1: INGESTING RAW DATA & EXTRACTING CONTINUOUS RECORD FEATURES")
print("="*70)

DATA_PATH_S1 = "student_resource/dataset/train/train_source1.tsv"
DATA_PATH_GT = "student_resource/dataset/train/train_ground_truth.tsv"

# Read representative sample with fixed seed for deterministic reproducibility
SAMPLE_SIZE = 50000
print(f"Loading {SAMPLE_SIZE} records from {DATA_PATH_S1}...")
df_s1 = pd.read_csv(DATA_PATH_S1, sep="\t", nrows=SAMPLE_SIZE)

print(f"Loading ground truth labels from {DATA_PATH_GT}...")
df_gt = pd.read_csv(DATA_PATH_GT, sep="\t")
gt_map = df_gt.set_index("source1_entity_id")["matched_entity_ids"]

def count_matches(val):
    if pd.isna(val) or str(val).strip() == "":
        return 0
    return len(str(val).split(","))

df_s1["target_match_count"] = df_s1["entity_id"].map(gt_map.map(count_matches)).fillna(0).astype(int)

# Extract raw structural & linguistic continuous features
df_s1["name_char_len"] = df_s1["business_name"].astype(str).str.len()
df_s1["address_char_len"] = df_s1["business_address"].astype(str).str.len()
df_s1["name_word_count"] = df_s1["business_name"].astype(str).str.split().str.len()
df_s1["address_word_count"] = df_s1["business_address"].astype(str).str.split().str.len()
df_s1["address_digit_count"] = df_s1["business_address"].astype(str).apply(lambda s: sum(c.isdigit() for c in s))

FEATURE_COLS = [
    "name_char_len",
    "address_char_len",
    "name_word_count",
    "address_word_count",
    "address_digit_count",
    "target_match_count"
]

X_raw = df_s1[FEATURE_COLS].copy()

# Raw Missing Value Check & Pure Median Imputation (if needed)
null_counts = X_raw.isnull().sum()
print("Missing Value Inspection across Raw Continuous Features:")
for col, cnt in null_counts.items():
    print(f"  - {col}: {cnt} nulls ({cnt/len(X_raw)*100:.2f}%)")
if null_counts.sum() > 0:
    print("Applying raw median imputation without scaling...")
    X_raw = X_raw.fillna(X_raw.median())
else:
    print("Dataset contains zero missing values in continuous numerical features.")

N = len(X_raw)
p = len(FEATURE_COLS)
print(f"Working Matrix Dimensions: {N} records x {p} continuous features.")

# ---------------------------------------------------------
# Step 2: Univariate Outlier Detection (IQR & Z-Score)
# ---------------------------------------------------------
print("\n" + "="*70)
print("STEP 2: COMPUTING UNIVARIATE OUTLIER FLAGS (IQR & Z-SCORE)")
print("="*70)

# 1. IQR Method
iqr_flags = pd.DataFrame(index=X_raw.index)
iqr_stats = {}
for col in FEATURE_COLS:
    q1 = X_raw[col].quantile(0.25)
    q3 = X_raw[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    flags = (X_raw[col] < lower_bound) | (X_raw[col] > upper_bound)
    iqr_flags[col] = flags
    iqr_stats[col] = {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "Lower": lower_bound, "Upper": upper_bound,
        "Outlier_Count": flags.sum(),
        "Outlier_Pct": (flags.sum() / N) * 100
    }
    print(f"[IQR] {col:<20}: Count={flags.sum():>5} ({flags.sum()/N*100:>5.2f}%) | Bounds=[{lower_bound:.2f}, {upper_bound:.2f}]")

union_iqr_series = iqr_flags.any(axis=1)
total_iqr_union = union_iqr_series.sum()
print(f"-> Total Univariate IQR Union Outliers (Distinct Rows): {total_iqr_union} ({total_iqr_union/N*100:.2f}%)")

# 2. Z-Score Method (|z| > 3.0)
z_flags = pd.DataFrame(index=X_raw.index)
z_stats = {}
for col in FEATURE_COLS:
    mu = X_raw[col].mean()
    sigma = X_raw[col].std(ddof=1)
    z = (X_raw[col] - mu) / sigma
    flags = z.abs() > 3.0
    z_flags[col] = flags
    z_stats[col] = {
        "Mean": mu, "Std": sigma,
        "Outlier_Count": flags.sum(),
        "Outlier_Pct": (flags.sum() / N) * 100
    }
    print(f"[Z-Score] {col:<20}: Count={flags.sum():>5} ({flags.sum()/N*100:>5.2f}%) | Mean={mu:.2f}, Std={sigma:.2f}")

union_z_series = z_flags.any(axis=1)
total_z_union = union_z_series.sum()
print(f"-> Total Univariate Z-Score Union Outliers (Distinct Rows): {total_z_union} ({total_z_union/N*100:.2f}%)")

# ---------------------------------------------------------
# Step 3: Multivariate Outlier Detection (Elliptical)
# ---------------------------------------------------------
print("\n" + "="*70)
print("STEP 3: COMPUTING MULTIVARIATE ELLIPTICAL OUTLIERS")
print("="*70)

# Method A: Mahalanobis Distance with Empirical Covariance and Chi-Square Critical Threshold
print("Fitting Empirical Covariance for Mahalanobis Distance...")
emp_cov = EmpiricalCovariance().fit(X_raw)
cov_inv = np.linalg.pinv(emp_cov.covariance_)
diff = X_raw.values - emp_cov.location_
md_sq_empirical = np.sum(np.dot(diff, cov_inv) * diff, axis=1)

# Chi-Square critical threshold at p < 0.001 (alpha = 0.001, critical quantile = 0.999), df = p
chi2_crit_p001 = stats.chi2.ppf(0.999, df=p)
mahalanobis_flags = md_sq_empirical > chi2_crit_p001
mahalanobis_count = int(mahalanobis_flags.sum())
print(f"Degrees of freedom: {p}, Alpha: 0.001, Chi-Square Critical Threshold: {chi2_crit_p001:.3f}")
print(f"-> Mahalanobis Distance Outliers (Chi^2 p<0.001): {mahalanobis_count} ({mahalanobis_count/N*100:.2f}%)")

# Method B: Robust Covariance via sklearn.covariance.EllipticEnvelope (FastMCD)
print("Fitting Robust Elliptic Envelope (contamination=0.01)...")
ee_model = EllipticEnvelope(contamination=0.01, random_state=RANDOM_STATE)
ee_model.fit(X_raw)
ee_flags = ee_model.predict(X_raw) == -1
ee_count = int(ee_flags.sum())
print(f"-> EllipticEnvelope (1% Contamination) Outliers: {ee_count} ({ee_count/N*100:.2f}%)")

# Method C: Robust Covariance Mahalanobis Distance (MinCovDet) with Chi-Square Critical Threshold
print("Fitting Robust Covariance (MinCovDet) with Chi-Square (p<0.001)...")
mcd = MinCovDet(random_state=RANDOM_STATE).fit(X_raw)
mcd_cov_inv = np.linalg.pinv(mcd.covariance_)
mcd_diff = X_raw.values - mcd.location_
md_sq_robust = np.sum(np.dot(mcd_diff, mcd_cov_inv) * mcd_diff, axis=1)
mcd_flags = md_sq_robust > chi2_crit_p001
mcd_count = int(mcd_flags.sum())
print(f"-> Robust MinCovDet Mahalanobis Outliers (Chi^2 p<0.001): {mcd_count} ({mcd_count/N*100:.2f}%)")

# ---------------------------------------------------------
# Step 4: Overlap / Intersection Analysis
# ---------------------------------------------------------
print("\n" + "="*70)
print("STEP 4: INTERSECTION & OVERLAP AUDIT BETWEEN METHODS")
print("="*70)

overlap_iqr_mahal = (union_iqr_series & mahalanobis_flags).sum()
overlap_z_mahal = (union_z_series & mahalanobis_flags).sum()
overlap_iqr_z = (union_iqr_series & union_z_series).sum()
overlap_all_three = (union_iqr_series & union_z_series & mahalanobis_flags).sum()
overlap_iqr_ee = (union_iqr_series & ee_flags).sum()
overlap_z_ee = (union_z_series & ee_flags).sum()

print(f"Intersection [Univariate IQR Union  AND  Univariate Z Union]: {overlap_iqr_z} ({overlap_iqr_z/total_iqr_union*100:.2f}% of IQR, {overlap_iqr_z/total_z_union*100:.2f}% of Z)")
print(f"Intersection [Univariate IQR Union  AND  Mahalanobis p<0.001]: {overlap_iqr_mahal} ({overlap_iqr_mahal/mahalanobis_count*100:.2f}% of Mahalanobis)")
print(f"Intersection [Univariate Z-Score Union  AND  Mahalanobis p<0.001]: {overlap_z_mahal} ({overlap_z_mahal/mahalanobis_count*100:.2f}% of Mahalanobis)")
print(f"Intersection [All Three (IQR Union & Z Union & Mahalanobis)]: {overlap_all_three} ({overlap_all_three/mahalanobis_count*100:.2f}% of Mahalanobis)")
print(f"Intersection [EllipticEnvelope (1%) AND Univariate IQR Union]: {overlap_iqr_ee} ({overlap_iqr_ee/ee_count*100:.2f}% of EE)")
print(f"Intersection [EllipticEnvelope (1%) AND Univariate Z-Score Union]: {overlap_z_ee} ({overlap_z_ee/ee_count*100:.2f}% of EE)")

# ---------------------------------------------------------
# Step 5: Distribution Profiling & Higher-Order Moments
# ---------------------------------------------------------
print("\n" + "="*70)
print("STEP 5: STATISTICAL PROFILING (MOMENTS, SKEWNESS, KURTOSIS)")
print("="*70)

profile_data = []
for col in FEATURE_COLS:
    vals = X_raw[col]
    mean_val = vals.mean()
    std_val = vals.std()
    median_val = vals.median()
    skew_val = vals.skew()
    kurt_val = vals.kurtosis()
    min_val = vals.min()
    max_val = vals.max()
    
    # Classifications
    skew_dir = "Right-skewed" if skew_val > 0.5 else ("Left-skewed" if skew_val < -0.5 else "Symmetric")
    tail_type = "Heavy-tailed (Leptokurtic)" if kurt_val > 1.0 else ("Light-tailed (Platykurtic)" if kurt_val < -1.0 else "Mesokurtic / Moderate")
    modality = "Unimodal" if col != "target_match_count" else "Multimodal (Discrete match spikes at 0, 1, 2, 3)"
    tail_pattern = "Continuous long tail" if max_val < 100 else "Extreme isolated outlier values"
    if col == "address_digit_count":
        tail_pattern = "Extreme isolated phone-number/tax-id injections in address"
    elif col == "target_match_count":
        tail_pattern = "Dense cluster [0, 4] with extreme hub entities up to 11"
        
    profile_data.append({
        "Feature": col,
        "Min": min_val,
        "Median": median_val,
        "Mean": mean_val,
        "Std": std_val,
        "Max": max_val,
        "Skewness (S)": skew_val,
        "Kurtosis (K)": kurt_val,
        "Modality": modality,
        "Skew Direction": skew_dir,
        "Tail Characteristics": tail_type,
        "Tail Pattern": tail_pattern
    })

profile_df = pd.DataFrame(profile_data)
print(profile_df[["Feature", "Min", "Median", "Mean", "Max", "Skewness (S)", "Kurtosis (K)"]].to_string(index=False))

# ---------------------------------------------------------
# Step 6: Task 2 Visualizations (KDE, Box, & Violin Plots)
# ---------------------------------------------------------
print("\n" + "="*70)
print("STEP 6: GENERATING HIGH-FIDELITY VISUALIZATIONS")
print("="*70)

# 1. KDE Plots
fig_kde, axes_kde = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig_kde.suptitle("Raw Feature Empirical Probability Densities (KDE) with Moment Annotations", fontsize=16, fontweight="bold", y=0.98)

palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

for idx, col in enumerate(FEATURE_COLS):
    ax = axes_kde[idx // 3, idx % 3]
    s_val = X_raw[col].skew()
    k_val = X_raw[col].kurtosis()
    mean_val = X_raw[col].mean()
    median_val = X_raw[col].median()
    
    # Plot KDE on raw scale (no transform, no clipping)
    sns.kdeplot(
        data=X_raw,
        x=col,
        ax=ax,
        color=palette[idx],
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        cut=0
    )
    
    # Reference vertical lines for Mean and Median to show skew separation
    ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.1f}")
    ax.axvline(median_val, color="black", linestyle=":", linewidth=1.5, label=f"Median: {median_val:.1f}")
    
    ax.set_title(f"{col}\n$S = {s_val:+.3f}$ | $K = {k_val:+.3f}$", fontsize=12, fontweight="semibold")
    ax.set_xlabel(f"Raw {col} Value", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
kde_filepath = os.path.join(OUTPUT_DIR, "kde_distributions.png")
fig_kde.savefig(kde_filepath, dpi=300, bbox_inches="tight")
plt.close(fig_kde)
print(f"Saved KDE figure to: {kde_filepath}")

# 2. Side-by-Side Box Plots and Violin Plots
fig_box_viol, axes_bv = plt.subplots(nrows=2, ncols=len(FEATURE_COLS), figsize=(22, 10))
fig_box_viol.suptitle("Raw Feature Dispersion & Extreme Tail Profiles: Box Plots (Top) vs. Violin Plots (Bottom)", fontsize=16, fontweight="bold", y=0.98)

for idx, col in enumerate(FEATURE_COLS):
    # Top Row: Box Plot showing IQR and individual extreme outliers
    ax_box = axes_bv[0, idx]
    sns.boxplot(
        y=X_raw[col],
        ax=ax_box,
        color=palette[idx],
        width=0.4,
        fliersize=3.5,
        flierprops={"marker": "o", "color": "darkred", "alpha": 0.5, "markeredgewidth": 0.5}
    )
    ax_box.set_title(f"Box Plot: {col}", fontsize=11, fontweight="semibold")
    ax_box.set_ylabel("Raw Value", fontsize=10)
    ax_box.grid(True, linestyle="--", alpha=0.5)
    
    # Bottom Row: Violin Plot highlighting multimodal density modes & tail thinning
    ax_viol = axes_bv[1, idx]
    sns.violinplot(
        y=X_raw[col],
        ax=ax_viol,
        color=palette[idx],
        inner="quartile",
        cut=0,
        linewidth=1.2
    )
    ax_viol.set_title(f"Violin Plot: {col}", fontsize=11, fontweight="semibold")
    ax_viol.set_ylabel("Raw Value", fontsize=10)
    ax_viol.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
box_viol_filepath = os.path.join(OUTPUT_DIR, "spread_box_violin.png")
fig_box_viol.savefig(box_viol_filepath, dpi=300, bbox_inches="tight")
plt.close(fig_box_viol)
print(f"Saved Box & Violin figure to: {box_viol_filepath}")

print("\n" + "="*70)
print("ANALYSIS EXECUTION COMPLETE & AUDIT READY.")
print("="*70)
