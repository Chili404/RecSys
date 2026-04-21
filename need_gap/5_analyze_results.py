import pandas as pd
from pathlib import Path
import json
import yaml
import numpy as np
from scipy.stats import wilcoxon


def clustered_bootstrap_ci(data, clusters, n_resamples=10000, ci=0.95, seed=42):
    """
    Compute percentile bootstrap CI for the mean, resampling at the cluster level.

    Resamples unique cluster IDs with replacement, then gathers all observations
    belonging to each sampled cluster. When each cluster has 1 observation,
    this is equivalent to a standard bootstrap.
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    clusters = np.asarray(clusters)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

    # Build a mapping from cluster ID to row indices
    cluster_to_idx = {}
    for i, c in enumerate(clusters):
        cluster_to_idx.setdefault(c, []).append(i)

    boot_means = np.empty(n_resamples)
    for b in range(n_resamples):
        sampled_clusters = unique_clusters[rng.integers(0, n_clusters, size=n_clusters)]
        indices = []
        for c in sampled_clusters:
            indices.extend(cluster_to_idx[c])
        boot_means[b] = data[indices].mean()

    alpha = (1 - ci) / 2
    lower = np.percentile(boot_means, 100 * alpha)
    upper = np.percentile(boot_means, 100 * (1 - alpha))
    return lower, upper


def cohens_d_paired(x, y):
    """Compute Cohen's d_z for paired data: mean(diff) / std(diff)."""
    diff = np.asarray(x) - np.asarray(y)
    if diff.std() == 0:
        return 0.0
    return diff.mean() / diff.std()


def benjamini_hochberg(p_values):
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.

    Returns adjusted p-values (q-values) in the same order as the input.
    """
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    # BH adjusted: p_adj[i] = min(p[i] * n / rank, 1.0), enforced to be monotone
    adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    # Enforce monotonicity (from right to left)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Put back in original order
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result

def validate_and_fix_reward_models(df):
    """
    Validate reward model scores and remove failed models.

    Checks for models that returned all zeros or all identical values (indicating failure),
    then recalculates means using only valid models.
    """
    # Load config to get model list
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    REWARD_MODELS = list(config['step_3']['reward_models'].keys())

    # Find valid models
    valid_models = []
    failed_models = []

    for model in REWARD_MODELS:
        pref_col = f'preference_reward_{model}'

        # Check if all values are identical (model failed)
        if pref_col in df.columns and df[pref_col].nunique() == 1:
            failed_models.append(model)
        else:
            valid_models.append(model)

    if len(failed_models) > 0:
        print(f"\n⚠️  WARNING: {len(failed_models)} model(s) failed (all identical scores):")
        for model in failed_models:
            print(f"  - {model}")
        print(f"\nRecalculating with {len(valid_models)} valid models only...")

        # Recalculate simple mean
        pref_cols = [f'preference_reward_{m}' for m in valid_models]
        need_cols = [f'need_aware_reward_{m}' for m in valid_models]

        df['preference_reward_mean'] = df[pref_cols].mean(axis=1)
        df['need_aware_reward_mean'] = df[need_cols].mean(axis=1)
        df['reward_gap'] = df['preference_reward_mean'] - df['need_aware_reward_mean']

        # Recalculate persona-weighted if it exists
        if 'person_weight' in df.columns:
            pref_weighted = []
            need_weighted = []

            for idx, row in df.iterrows():
                # Get persona weights for valid models only
                person_weight_full = np.array(row['person_weight'])[:8]
                valid_indices = [i for i, m in enumerate(REWARD_MODELS) if m in valid_models]
                person_weight = person_weight_full[valid_indices]

                # Renormalize to sum to 1
                person_weight = person_weight / person_weight.sum()

                # Get scores
                pref_scores = [row[f'preference_reward_{m}'] for m in valid_models]
                need_scores = [row[f'need_aware_reward_{m}'] for m in valid_models]

                # Weighted sum
                pref_weighted.append(np.dot(person_weight, pref_scores))
                need_weighted.append(np.dot(person_weight, need_scores))

            df['preference_reward_persona_weighted'] = pref_weighted
            df['need_aware_reward_persona_weighted'] = need_weighted
            df['reward_gap_persona_weighted'] = df['preference_reward_persona_weighted'] - df['need_aware_reward_persona_weighted']

        print(f"✓ Recalculated using {len(valid_models)} models: {', '.join(valid_models)}\n")

    return df

def load_scored_responses():
    """Load scored responses DataFrame"""
    fully_scored_path = Path(__file__).parent / "data" / "fully_scored_responses.parquet"

    if fully_scored_path.exists():
        df = pd.read_parquet(fully_scored_path)
        print(f"Loaded {len(df)} fully scored responses (with S_need)")
        df = validate_and_fix_reward_models(df)
        return df
    else:
        print("ERROR: fully_scored_responses.parquet not found")
        print("Please run 4_score_need_alignment_async.py first!")
        return None

def _build_table_2_rows(subset_df, subset_name, has_s_need):
    """Build Table 2 rows for a given subset with clustered bootstrap CIs, Wilcoxon p-values, and Cohen's d."""
    pref_rpref = subset_df['preference_reward_mean'].values
    need_rpref = subset_df['need_aware_reward_mean'].values
    clusters = subset_df['person_id'].values

    # Clustered bootstrap CIs for R_pref means
    pref_rpref_ci = clustered_bootstrap_ci(pref_rpref, clusters)
    need_rpref_ci = clustered_bootstrap_ci(need_rpref, clusters)

    # Wilcoxon signed-rank test for R_pref
    _, rpref_p = wilcoxon(pref_rpref, need_rpref)
    rpref_d = cohens_d_paired(pref_rpref, need_rpref)

    # Store raw p-values for later BH correction
    raw_p_values = [rpref_p]

    pref_row = {
        'Subset': subset_name,
        'Response Type': 'Preference-matched',
        'R_pref': pref_rpref.mean(),
        'R_pref_CI': f"({pref_rpref_ci[0]:.4f}, {pref_rpref_ci[1]:.4f})",
        'n': len(subset_df),
        'R_pref_p': rpref_p,
        'R_pref_d': rpref_d,
    }

    need_row = {
        'Subset': subset_name,
        'Response Type': 'Need-aware',
        'R_pref': need_rpref.mean(),
        'R_pref_CI': f"({need_rpref_ci[0]:.4f}, {need_rpref_ci[1]:.4f})",
        'n': len(subset_df),
        'R_pref_p': '',
        'R_pref_d': '',
    }

    if has_s_need:
        pref_sneed = subset_df['pref_s_need'].values
        need_sneed = subset_df['need_s_need'].values

        pref_sneed_ci = clustered_bootstrap_ci(pref_sneed, clusters)
        need_sneed_ci = clustered_bootstrap_ci(need_sneed, clusters)

        _, sneed_p = wilcoxon(pref_sneed, need_sneed)
        sneed_d = cohens_d_paired(pref_sneed, need_sneed)

        raw_p_values.append(sneed_p)

        pref_row['S_need'] = pref_sneed.mean()
        pref_row['S_need_CI'] = f"({pref_sneed_ci[0]:.4f}, {pref_sneed_ci[1]:.4f})"
        pref_row['S_need_p'] = sneed_p
        pref_row['S_need_d'] = sneed_d

        need_row['S_need'] = need_sneed.mean()
        need_row['S_need_CI'] = f"({need_sneed_ci[0]:.4f}, {need_sneed_ci[1]:.4f})"
        need_row['S_need_p'] = ''
        need_row['S_need_d'] = ''
    else:
        pref_row['S_need'] = 'TBD'
        pref_row['S_need_CI'] = ''
        pref_row['S_need_p'] = ''
        pref_row['S_need_d'] = ''
        need_row['S_need'] = 'TBD'
        need_row['S_need_CI'] = ''
        need_row['S_need_p'] = ''
        need_row['S_need_d'] = ''

    return pref_row, need_row, raw_p_values


def generate_table_2(df):
    """
    Generate Table 2: Preference reward vs. need-alignment on PersonalLLM benchmark.

    Shows all 4 combinations with Wilcoxon p-values (BH-corrected), clustered bootstrap
    95% CIs, and Cohen's d.
    """
    print("\n" + "="*80)
    print("Table 2: Preference Reward vs. Need-Alignment")
    print("="*80)

    divergent_df = df[df['divergence_type'] != 'none']
    non_divergent_df = df[df['divergence_type'] == 'none']
    has_s_need = 'pref_s_need' in df.columns and 'need_s_need' in df.columns

    table_2_data = []
    all_raw_p = []
    # Track which rows and columns hold p-values for later replacement
    p_locations = []  # (row_index, column_name)

    if len(divergent_df) > 0:
        pref_row, need_row, raw_ps = _build_table_2_rows(divergent_df, 'Divergent', has_s_need)
        row_idx = len(table_2_data)
        table_2_data.extend([pref_row, need_row])
        all_raw_p.extend(raw_ps)
        p_locations.append((row_idx, 'R_pref_p'))
        if has_s_need:
            p_locations.append((row_idx, 'S_need_p'))

    if len(non_divergent_df) > 0:
        pref_row, need_row, raw_ps = _build_table_2_rows(non_divergent_df, 'Non-divergent', has_s_need)
        row_idx = len(table_2_data)
        table_2_data.extend([pref_row, need_row])
        all_raw_p.extend(raw_ps)
        p_locations.append((row_idx, 'R_pref_p'))
        if has_s_need:
            p_locations.append((row_idx, 'S_need_p'))

    # Apply BH-FDR correction across all p-values in this table
    adjusted_p = benjamini_hochberg(all_raw_p)
    for i, (row_idx, col) in enumerate(p_locations):
        table_2_data[row_idx][col] = adjusted_p[i]

    table_2_df = pd.DataFrame(table_2_data)

    print("\nBreakdown:")
    print(f"  Divergent cases: {len(divergent_df)}")
    print(f"  Non-divergent cases: {len(non_divergent_df)}")
    print(f"  Total: {len(df)}\n")

    print(table_2_df.to_string(index=False))

    print("\n--- Statistical Notes ---")
    print("  p-values: Wilcoxon signed-rank test (paired, two-sided), BH-FDR corrected")
    print("  CIs: 95% clustered bootstrap (10,000 resamples, clustered by person_id)")
    print("  d: Cohen's d_z for paired effect size")
    print("  p-values and d reported on the Preference-matched row for each subset")

    return table_2_df

def generate_table_3(df):
    """
    Generate Table 3: Need-alignment gap by divergence type.

    ΔS_need = need_aware S_need - preference_matched S_need
    With clustered bootstrap 95% CIs, Wilcoxon p-values (BH-corrected), and Cohen's d_z.
    """
    print("\n" + "="*80)
    print("Table 3: Need-Alignment Gap by Divergence Type")
    print("="*80)

    has_s_need = 'pref_s_need' in df.columns and 'need_s_need' in df.columns

    table_3_data = []
    all_raw_p = []
    # Track (row_index, column_name) for each raw p-value
    p_locations = []

    for div_type in sorted(df['divergence_type'].unique()):
        subset = df[df['divergence_type'] == div_type]
        clusters = subset['person_id'].values

        pref_rpref = subset['preference_reward_mean'].values
        need_rpref = subset['need_aware_reward_mean'].values
        delta_rpref = need_rpref - pref_rpref

        delta_rpref_ci = clustered_bootstrap_ci(delta_rpref, clusters)
        delta_rpref_d = cohens_d_paired(need_rpref, pref_rpref)
        _, rpref_p = wilcoxon(pref_rpref, need_rpref)

        row_idx = len(table_3_data)
        all_raw_p.append(rpref_p)
        p_locations.append((row_idx, 'ΔR_pref_p'))

        row = {
            'Divergence Type': div_type.replace('_', ' ').title(),
            'ΔR_pref': delta_rpref.mean(),
            'ΔR_pref_CI': f"({delta_rpref_ci[0]:.4f}, {delta_rpref_ci[1]:.4f})",
            'ΔR_pref_p': rpref_p,
            'ΔR_pref_d': delta_rpref_d,
            'n': len(subset),
        }

        if has_s_need:
            pref_sneed = subset['pref_s_need'].values
            need_sneed = subset['need_s_need'].values
            delta_sneed = need_sneed - pref_sneed

            delta_sneed_ci = clustered_bootstrap_ci(delta_sneed, clusters)
            delta_sneed_d = cohens_d_paired(need_sneed, pref_sneed)
            _, sneed_p = wilcoxon(pref_sneed, need_sneed)

            all_raw_p.append(sneed_p)
            p_locations.append((row_idx, 'ΔS_need_p'))

            row['ΔS_need'] = delta_sneed.mean()
            row['ΔS_need_CI'] = f"({delta_sneed_ci[0]:.4f}, {delta_sneed_ci[1]:.4f})"
            row['ΔS_need_p'] = sneed_p
            row['ΔS_need_d'] = delta_sneed_d
        else:
            row['ΔS_need'] = 'TBD'
            row['ΔS_need_CI'] = ''
            row['ΔS_need_p'] = ''
            row['ΔS_need_d'] = ''

        table_3_data.append(row)

    # Apply BH-FDR correction across all p-values in this table
    adjusted_p = benjamini_hochberg(all_raw_p)
    for i, (row_idx, col) in enumerate(p_locations):
        table_3_data[row_idx][col] = adjusted_p[i]

    table_3_df = pd.DataFrame(table_3_data)
    print(table_3_df.to_string(index=False))

    print("\n--- Statistical Notes ---")
    print("  p-values: Wilcoxon signed-rank test (paired, two-sided), BH-FDR corrected")
    print("  CIs: 95% clustered bootstrap (10,000 resamples, clustered by person_id)")
    print("  d: Cohen's d_z for paired effect size")

    return table_3_df

def generate_table_4(df):
    """
    Generate Table 4: Qualitative examples showing query-need divergence.

    Selects one example from each divergence type showing the contrast
    between preference-matched and need-aware responses.
    """
    print("\n" + "="*80)
    print("Table 4: Qualitative Examples")
    print("="*80)

    examples = []

    for div_type in sorted(df['divergence_type'].unique()):
        subset = df[df['divergence_type'] == div_type]

        # Select example with largest reward gap
        example_idx = subset['reward_gap'].abs().idxmax()
        example = subset.loc[example_idx]

        examples.append({
            'divergence_type': div_type,
            'prompt': example['test_prompt'],
            'preference_matched_response': example['preference_matched_response'],
            'need_aware_response': example['need_aware_response'],
            'preference_reward': example['preference_reward_mean'],
            'need_aware_reward': example['need_aware_reward_mean'],
            'need_aware_analysis': example['need_aware_analysis']
        })

    examples_path = Path(__file__).parent / "results" / "qualitative_examples.json"
    examples_path.parent.mkdir(exist_ok=True)

    with open(examples_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"Saved qualitative examples to: {examples_path}")

    for i, ex in enumerate(examples):
        print(f"\n--- Example {i+1}: {ex['divergence_type']} ---")
        print(f"Prompt: {ex['prompt'][:150]}...")
        print(f"Preference reward: {ex['preference_reward']:.3f}")
        print(f"Need-aware reward: {ex['need_aware_reward']:.3f}")

    return examples

def generate_top_discrepancies(df, top_n_per_category=3):
    """
    Generate top N cases per divergence category where R_pref and S_need most disagree.

    Discrepancy = cases where reward models prefer one response but
    need-alignment judge prefers the other.
    """
    print("\n" + "="*80)
    print(f"Top {top_n_per_category} Discrepancies per Divergence Type (R_pref vs S_need)")
    print("="*80)

    # Check if S_need scores are available
    has_s_need = 'pref_s_need' in df.columns and 'need_s_need' in df.columns

    if not has_s_need:
        print("S_need scores not available. Run step 5 first.")
        return []

    # Calculate discrepancy
    # Positive reward_gap = pref wins on rewards
    # Positive s_need_gap = need wins on alignment
    # High discrepancy = opposite directions or large magnitude difference
    df['reward_gap_normalized'] = (df['preference_reward_mean'] - df['need_aware_reward_mean'])
    df['s_need_gap'] = (df['need_s_need'] - df['pref_s_need'])

    # Discrepancy metric: where they disagree most
    # reward_gap is always positive (pref wins), s_need_gap can be positive or negative
    # We want cases where s_need_gap is large and positive (need wins) despite reward gap
    df['discrepancy'] = df['s_need_gap'] - (df['reward_gap_normalized'] / df['reward_gap_normalized'].abs().max())

    # Get top N per divergence type
    all_examples = []
    for div_type in sorted(df['divergence_type'].unique()):
        subset = df[df['divergence_type'] == div_type]
        top_for_type = subset.nlargest(top_n_per_category, 'discrepancy')

        for idx, row in top_for_type.iterrows():
            all_examples.append({
                'prompt': row['test_prompt'],
                'divergence_type': row['divergence_type'],
                'person_id': row['person_id'],

                # Full responses
                'preference_matched_response': row['preference_matched_response'],
                'need_aware_response': row['need_aware_response'],
                'need_aware_analysis': row['need_aware_analysis'],

                # R_pref scores
                'preference_matched_rpref': float(row['preference_reward_mean']),
                'need_aware_rpref': float(row['need_aware_reward_mean']),
                'rpref_gap': float(row['reward_gap']),

                # S_need scores
                'preference_matched_sneed': float(row['pref_s_need']),
                'need_aware_sneed': float(row['need_s_need']),
                'sneed_gap': float(row['s_need_gap']),

                # Discrepancy metric
                'discrepancy_score': float(row['discrepancy'])
            })

    examples = all_examples

    # Save to JSON
    discrepancy_path = Path(__file__).parent / "results" / "top_discrepancies.json"
    discrepancy_path.parent.mkdir(exist_ok=True)

    with open(discrepancy_path, 'w') as f:
        json.dump(examples, f, indent=2)

    print(f"\nSaved top {top_n_per_category} discrepancies per category ({len(examples)} total) to: {discrepancy_path}")
    print("\nBreakdown by divergence type:")
    from collections import Counter
    type_counts = Counter(ex['divergence_type'] for ex in examples)
    for dtype, count in sorted(type_counts.items()):
        print(f"  {dtype}: {count} cases")

    print("\nSample preview (first 3):")
    for i, ex in enumerate(examples[:3]):
        print(f"\n{i+1}. {ex['divergence_type']}")
        print(f"   R_pref gap: {ex['rpref_gap']:+.3f} (pref wins)")
        print(f"   S_need gap: {ex['sneed_gap']:+.3f} (need wins)" if ex['sneed_gap'] > 0 else f"   S_need gap: {ex['sneed_gap']:+.3f} (pref wins)")
        print(f"   Discrepancy: {ex['discrepancy_score']:.3f}")
        print(f"   Prompt: {ex['prompt'][:100]}...")

    return examples

def generate_judge_validation_summary():
    """
    Load and display judge validation results from step 4.5.

    Returns the validation dict, or None if results not found.
    """
    validation_path = Path(__file__).parent / "results" / "judge_validation.json"
    if not validation_path.exists():
        print("\n[Judge validation not yet run — skipping. Run 4.5_validate_judge.py first.]")
        return None

    with open(validation_path, 'r') as f:
        val = json.load(f)

    print("\n" + "="*80)
    print("S_need Judge Validation (Human Ground Truth Calibration)")
    print("="*80)

    t1 = val["tier_1_aita_yta_sycophancy"]
    t2 = val["tier_2_ideal_response"]
    summary = val["summary"]

    # Tier 1
    print(f"\n--- Tier 1: AITA-YTA Sycophancy Detection (n={t1['n_valid']}) ---")
    print(f"  Human label: {t1['human_label_semantics']}")
    print(f"  Label distribution: {t1['human_label_distribution']}")
    print(f"\n  ROC-AUC (appropriate_redirection): {t1['roc_auc_appropriate_redirection']:.4f}")
    print(f"  ROC-AUC (s_need):                  {t1['roc_auc_s_need']:.4f}")
    print(f"  Point-biserial r (s_need):          {t1['point_biserial_s_need']['r']:.4f} "
          f"(p={t1['point_biserial_s_need']['p']:.2e})")

    # Bucketed agreement
    buckets = t1["bucketed_agreement"]
    print(f"\n  Bucketed agreement (judge redirection tertiles -> human challenge rate):")
    for bucket_name, bucket_data in buckets.items():
        print(f"    {bucket_name}: {bucket_data['human_challenges_rate']:.1%} "
              f"(n={bucket_data['n']})")

    # Tier 2
    print(f"\n--- Tier 2: Ideal Response Recognition (n={t2['n_valid']}) ---")
    print(f"  Mean S_need for ideal responses: {t2['mean_s_need']:.4f} (std={t2['std_s_need']:.4f})")
    print(f"  Quality check: {t2['ideal_response_quality_check']}")

    # Overall verdict
    print(f"\n--- Validation Verdict ---")
    if summary["judge_discriminates_sycophancy"]:
        print(f"  [PASS] Judge discriminates sycophancy (AUC={summary['tier_1_auc_sneed']:.3f} > 0.5)")
    else:
        print(f"  [WARN] Judge does not discriminate sycophancy (AUC={summary['tier_1_auc_sneed']:.3f})")

    if summary["judge_recognizes_ideal_responses"]:
        print(f"  [PASS] Judge recognizes ideal responses (mean S_need={summary['tier_2_mean_s_need']:.3f} > 0.7)")
    else:
        print(f"  [WARN] Judge does not recognize ideal responses (mean S_need={summary['tier_2_mean_s_need']:.3f})")

    return val


def generate_statistics_summary(df):
    """Generate overall statistics summary"""
    print("\n" + "="*80)
    print("Overall Statistics Summary")
    print("="*80)

    print(f"\nTotal responses analyzed: {len(df)}")

    print("\nDivergence type distribution:")
    print(df['divergence_type'].value_counts())

    print("\nReward scores:")
    print("  Preference-matched (R_pref):")
    print(f"    Mean: {df['preference_reward_mean'].mean():.4f}")
    print(f"    Std:  {df['preference_reward_mean'].std():.4f}")

    print("\n  Need-aware (R_pref):")
    print(f"    Mean: {df['need_aware_reward_mean'].mean():.4f}")
    print(f"    Std:  {df['need_aware_reward_mean'].std():.4f}")

    print("\n  Reward gap (pref - need):")
    print(f"    Mean: {df['reward_gap'].mean():.4f}")
    print(f"    Std:  {df['reward_gap'].std():.4f}")

    # Percentage of cases where preference-matched scores higher
    pref_wins = (df['preference_reward_mean'] > df['need_aware_reward_mean']).sum()
    pref_win_pct = 100 * pref_wins / len(df)

    print(f"\n  Preference-matched scores higher: {pref_wins}/{len(df)} ({pref_win_pct:.1f}%)")

    # Analysis by divergence type
    print("\nReward gap by divergence type:")
    for div_type in sorted(df['divergence_type'].unique()):
        subset = df[df['divergence_type'] == div_type]
        mean_gap = subset['reward_gap'].mean()
        print(f"  {div_type}: {mean_gap:+.4f} (n={len(subset)})")

    # Data source breakdown
    if 'data_source' in df.columns:
        print("\nData source distribution:")
        print(df['data_source'].value_counts())
        print("\nDivergence type by data source:")
        print(df.groupby(['data_source', 'divergence_type']).size().unstack(fill_value=0))


def main():
    """Main analysis pipeline"""
    # Load data
    df = load_scored_responses()
    if df is None:
        return

    # Generate statistics summary
    generate_statistics_summary(df)

    # Generate tables
    table_2_df = generate_table_2(df)
    table_3_df = generate_table_3(df)

    # Generate top discrepancies (R_pref vs S_need mismatches)
    # Get top 3 cases per divergence type
    generate_top_discrepancies(df, top_n_per_category=3)

    # Judge validation summary (from step 4.5)
    validation = generate_judge_validation_summary()

    # Save tables
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    table_2_df.to_csv(results_dir / "table_2.csv", index=False)
    table_3_df.to_csv(results_dir / "table_3.csv", index=False)

    # Save statistical methods metadata
    stats_meta = {
        "column_definitions": {
            "_p columns (R_pref_p, S_need_p, ΔR_pref_p, ΔS_need_p)": {
                "test": "Wilcoxon signed-rank test (paired, two-sided)",
                "correction": "Benjamini-Hochberg FDR correction applied across all p-values within each table"
            },
            "_d columns (R_pref_d, S_need_d, ΔR_pref_d, ΔS_need_d)": {
                "test": "Cohen's d_z for paired data",
                "formula": "mean(diff) / std(diff)"
            },
            "_CI columns (R_pref_CI, S_need_CI, ΔR_pref_CI, ΔS_need_CI)": {
                "method": "Percentile bootstrap",
                "n_resamples": 10000,
                "confidence_level": 0.95,
                "clustering": "Resampled at person_id level (clustered bootstrap)",
                "seed": 42
            }
        },
        "data": {
            "n_total": len(df),
            "n_divergent": len(df[df['divergence_type'] != 'none']),
            "n_non_divergent": len(df[df['divergence_type'] == 'none']),
            "divergence_types": df['divergence_type'].value_counts().to_dict(),
            "n_unique_persons": int(df['person_id'].nunique())
        }
    }

    # Include judge validation summary if available
    if validation is not None:
        stats_meta["judge_validation"] = {
            "tier_1_aita_yta": {
                "n": validation["tier_1_aita_yta_sycophancy"]["n_valid"],
                "roc_auc_s_need": validation["tier_1_aita_yta_sycophancy"]["roc_auc_s_need"],
                "roc_auc_appropriate_redirection": validation["tier_1_aita_yta_sycophancy"]["roc_auc_appropriate_redirection"],
                "point_biserial_s_need": validation["tier_1_aita_yta_sycophancy"]["point_biserial_s_need"],
            },
            "tier_2_ideal_response": {
                "n": validation["tier_2_ideal_response"]["n_valid"],
                "mean_s_need": validation["tier_2_ideal_response"]["mean_s_need"],
                "quality_check": validation["tier_2_ideal_response"]["ideal_response_quality_check"],
            },
            "verdict": {
                "judge_discriminates_sycophancy": validation["summary"]["judge_discriminates_sycophancy"],
                "judge_recognizes_ideal_responses": validation["summary"]["judge_recognizes_ideal_responses"],
            }
        }

    with open(results_dir / "statistical_methods.json", 'w') as f:
        json.dump(stats_meta, f, indent=2)
    print(f"\nSaved statistical methods metadata to: {results_dir / 'statistical_methods.json'}")

    print(f"Saved CSV tables to: {results_dir}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)

if __name__ == "__main__":
    main()
