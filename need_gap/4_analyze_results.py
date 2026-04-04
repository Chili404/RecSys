import pandas as pd
from pathlib import Path
import json

def load_scored_responses():
    """Load scored responses DataFrame"""
    fully_scored_path = Path(__file__).parent / "data" / "fully_scored_responses.parquet"
    scored_path = Path(__file__).parent / "data" / "scored_responses.parquet"

    if fully_scored_path.exists():
        df = pd.read_parquet(fully_scored_path)
        print(f"Loaded {len(df)} fully scored responses (with S_need)")
        return df
    elif scored_path.exists():
        df = pd.read_parquet(scored_path)
        print(f"Loaded {len(df)} scored responses (without S_need)")
        return df
    else:
        print("ERROR: No scored responses found")
        print("Please run 3_score_responses.py first!")
        return None

def generate_table_2(df):
    """
    Generate Table 2: Preference reward vs. need-alignment on PersonalLLM benchmark.

    Shows all 4 combinations:
    - Divergent + Preference-matched
    - Divergent + Need-aware
    - Non-divergent + Preference-matched
    - Non-divergent + Need-aware
    """
    print("\n" + "="*80)
    print("Table 2: Preference Reward vs. Need-Alignment")
    print("="*80)

    # Properly separate divergent and non-divergent cases
    # Divergent: divergence_type != 'none'
    # Non-divergent: divergence_type == 'none'
    divergent_df = df[df['divergence_type'] != 'none']
    non_divergent_df = df[df['divergence_type'] == 'none']

    # Check if S_need scores are available
    has_s_need = 'pref_s_need' in df.columns and 'need_s_need' in df.columns

    table_2_data = []

    # Divergent cases
    if len(divergent_df) > 0:
        table_2_data.append({
            'Subset': 'Divergent',
            'Response Type': 'Preference-matched',
            'R_pref': divergent_df['preference_reward_mean'].mean(),
            'S_need': divergent_df['pref_s_need'].mean() if has_s_need else 'TBD',
            'n': len(divergent_df)
        })

        table_2_data.append({
            'Subset': 'Divergent',
            'Response Type': 'Need-aware',
            'R_pref': divergent_df['need_aware_reward_mean'].mean(),
            'S_need': divergent_df['need_s_need'].mean() if has_s_need else 'TBD',
            'n': len(divergent_df)
        })

    # Non-divergent cases
    if len(non_divergent_df) > 0:
        table_2_data.append({
            'Subset': 'Non-divergent',
            'Response Type': 'Preference-matched',
            'R_pref': non_divergent_df['preference_reward_mean'].mean(),
            'S_need': non_divergent_df['pref_s_need'].mean() if has_s_need else 'TBD',
            'n': len(non_divergent_df)
        })

        table_2_data.append({
            'Subset': 'Non-divergent',
            'Response Type': 'Need-aware',
            'R_pref': non_divergent_df['need_aware_reward_mean'].mean(),
            'S_need': non_divergent_df['need_s_need'].mean() if has_s_need else 'TBD',
            'n': len(non_divergent_df)
        })

    table_2_df = pd.DataFrame(table_2_data)

    print(f"\nBreakdown:")
    print(f"  Divergent cases: {len(divergent_df)}")
    print(f"  Non-divergent cases: {len(non_divergent_df)}")
    print(f"  Total: {len(df)}\n")

    print(table_2_df.to_string(index=False))

    return table_2_df

def generate_table_3(df):
    """
    Generate Table 3: Need-alignment gap by divergence type.

    ΔS_need = need_aware S_need - preference_matched S_need
    """
    print("\n" + "="*80)
    print("Table 3: Need-Alignment Gap by Divergence Type")
    print("="*80)

    # Check if S_need scores are available
    has_s_need = 'pref_s_need' in df.columns and 'need_s_need' in df.columns

    table_3_data = []

    for div_type in sorted(df['divergence_type'].unique()):
        subset = df[df['divergence_type'] == div_type]

        delta_s_need = (subset['need_s_need'].mean() - subset['pref_s_need'].mean()) if has_s_need else 'TBD'

        table_3_data.append({
            'Divergence Type': div_type.replace('_', ' ').title(),
            'ΔR_pref': subset['need_aware_reward_mean'].mean() - subset['preference_reward_mean'].mean(),
            'ΔS_need': delta_s_need,
            'n': len(subset)
        })

    table_3_df = pd.DataFrame(table_3_data)
    print(table_3_df.to_string(index=False))

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
            'preference_matched_response': example['preference_matched_response'][:300] + "...",
            'need_aware_response': example['need_aware_response'][:300] + "...",
            'preference_reward': example['preference_reward_mean'],
            'need_aware_reward': example['need_aware_reward_mean'],
            'need_aware_analysis': example['need_aware_analysis'][:200] + "..."
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

    # Save tables
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    table_2_df.to_csv(results_dir / "table_2.csv", index=False)
    table_3_df.to_csv(results_dir / "table_3.csv", index=False)

    print(f"\nSaved CSV tables to: {results_dir}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print("\nNOTE: S_need scores are marked as 'TBD' until the need-alignment judge is implemented.")
    print("Once you have the judge, run this script again to populate those fields.")

if __name__ == "__main__":
    main()
