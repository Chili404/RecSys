"""
Step 6: Cross-Model Need Gap Analysis (Table 4).

Scores 11 production LLM responses from the AITA-YTA bench with the S_need judge,
producing per-model statistics and pairwise comparisons. Extends the need gap
claim from PersonalLLM (Step 5) to be systemic across production models.

Stages (with checkpoint/resume):
  1. Load & stratified sample 250 rows from NeedGap-Bench.xlsx
  2. Score all (prompt, model) pairs with GPT-4o S_need judge (~2750 API calls)
  3. Merge R_pref if companion notebook output exists
  4. Compute per-model stats, pairwise tests, generate Table 4
"""

import asyncio
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.stats import wilcoxon, spearmanr, pointbiserialr
from tqdm.asyncio import tqdm as async_tqdm
import os


# ─── Config / prompt loading ────────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_judge_prompt(config):
    prompt_path = Path(__file__).parent / config["step_4"]["need_alignment_judge"]["prompt_path"]
    with open(prompt_path, "r") as f:
        return f.read()


# ─── Async scoring (reused from 4.5_validate_judge.py) ──────────────────────

async def score_with_judge_async(client, prompt, divergence_type, response, judge_template, semaphore, model="gpt-4o", temperature=0.3):
    """Score a single response with the need-alignment judge."""
    prompt_str = str(prompt) if prompt is not None else ""
    divergence_type_str = str(divergence_type) if divergence_type is not None else "unknown"
    response_str = str(response) if response is not None else ""

    filled_prompt = (
        judge_template
        .replace("{prompt}", prompt_str)
        .replace("{divergence_type}", divergence_type_str)
        .replace("{response}", response_str)
    )

    try:
        async with semaphore:
            result = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator of AI response quality for need-alignment."},
                    {"role": "user", "content": filled_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )

        scores = json.loads(result.choices[0].message.content)

        s_need = (
            scores["need_fulfillment"]
            + scores["expert_approval"]
            + scores["appropriate_redirection"]
        ) / 15.0

        return {
            "need_fulfillment": scores["need_fulfillment"],
            "expert_approval": scores["expert_approval"],
            "appropriate_redirection": scores["appropriate_redirection"],
            "s_need": s_need,
            "error": None,
        }

    except Exception as e:
        return {
            "need_fulfillment": None,
            "expert_approval": None,
            "appropriate_redirection": None,
            "s_need": None,
            "error": str(e),
        }


# ─── Statistical utilities (reused from 5_analyze_results.py) ───────────────

def compute_roc_auc(y_true, y_scores):
    """Compute ROC-AUC via the trapezoidal rule."""
    y_true = np.asarray(y_true, dtype=int)
    y_scores = np.asarray(y_scores, dtype=float)

    desc_order = np.argsort(-y_scores)
    y_true_sorted = y_true[desc_order]
    y_scores_sorted = y_scores[desc_order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tpr_prev, fpr_prev = 0.0, 0.0
    auc = 0.0
    tp, fp = 0, 0

    prev_score = None
    for i in range(len(y_true_sorted)):
        if y_scores_sorted[i] != prev_score and prev_score is not None:
            tpr = tp / n_pos
            fpr = fp / n_neg
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
            tpr_prev, fpr_prev = tpr, fpr

        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prev_score = y_scores_sorted[i]

    tpr = tp / n_pos
    fpr = fp / n_neg
    auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0

    return auc


def clustered_bootstrap_ci(data, clusters, n_resamples=10000, ci=0.95, seed=42):
    """Compute percentile bootstrap CI for the mean, resampling at the cluster level."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    clusters = np.asarray(clusters)

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)

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
    """Apply Benjamini-Hochberg FDR correction to a list of p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]

    adjusted = np.minimum(sorted_p * n / np.arange(1, n + 1), 1.0)
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


# ─── Stage 1: Load & Stratified Sample ──────────────────────────────────────

def load_bench_data(config):
    """Load the AITA-YTA bench Excel and deduplicate Mistral-7B columns."""
    cfg = config["step_6"]
    bench_path = Path(__file__).parent / cfg["bench_path"]
    df = pd.read_excel(bench_path, sheet_name=cfg["sheet"])
    print(f"Loaded {len(df)} rows from sheet '{cfg['sheet']}'")

    # Deduplicate Mistral-7B columns (known issue: duplicate columns in Excel)
    mistral_cols = [c for c in df.columns if "mistral-7b" in c.lower() or "mistral_7b" in c.lower()]
    if len(mistral_cols) > 1:
        print(f"  Found {len(mistral_cols)} Mistral-7B columns: {mistral_cols}")
        canonical = mistral_cols[0]
        to_drop = mistral_cols[1:]
        df = df.drop(columns=to_drop)
        if canonical != "Mistral-7B":
            df = df.rename(columns={canonical: "Mistral-7B"})
        print(f"  Kept '{canonical}' as 'Mistral-7B', dropped {to_drop}")

    return df


def stratified_sample(df, config):
    """Stratified sample of n rows preserving class balance of validation_human."""
    cfg = config["step_6"]
    n = cfg["sample_size"]
    seed = cfg["random_seed"]
    label_col = cfg["human_label_col"]

    # Drop rows missing the label
    df_valid = df.dropna(subset=[label_col]).copy()
    print(f"  Rows with valid '{label_col}': {len(df_valid)}")

    # Compute class proportions and allocate counts
    class_counts = df_valid[label_col].value_counts()
    total = len(df_valid)
    proportions = class_counts / total

    # Largest-remainder allocation
    raw_alloc = proportions * n
    floor_alloc = raw_alloc.apply(math.floor)
    remainder = raw_alloc - floor_alloc
    shortfall = int(n - floor_alloc.sum())

    # Distribute remainder to classes with largest fractional parts
    top_classes = remainder.sort_values(ascending=False).index[:shortfall]
    for cls in top_classes:
        floor_alloc[cls] += 1

    alloc = floor_alloc.astype(int)
    print(f"  Stratified allocation (n={n}): {alloc.to_dict()}")

    # Sample from each class
    sampled_parts = []
    for cls, count in alloc.items():
        class_df = df_valid[df_valid[label_col] == cls]
        sampled = class_df.sample(n=count, random_state=seed)
        sampled_parts.append(sampled)

    sampled_df = pd.concat(sampled_parts).reset_index(drop=True)

    # Add a prompt_id for joining
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df["prompt_id"] = range(len(sampled_df))

    print(f"  Final sample: {len(sampled_df)} rows")
    return sampled_df


# ─── Stage 2: S_need Scoring ────────────────────────────────────────────────

async def score_all_models_sneed(client, df, config, judge_template):
    """Score all (prompt, model) pairs with the S_need judge. Resumable."""
    cfg = config["step_6"]
    model_columns = cfg["model_columns"]
    divergence_type = cfg["divergence_type"]
    prompt_col = cfg["prompt_col"]
    judge_cfg = cfg["judge"]

    semaphore = asyncio.Semaphore(judge_cfg["max_concurrent"])

    # Build list of (prompt_id, model, prompt_text, response_text) pairs to score
    pairs_to_score = []
    for model_col in model_columns:
        if model_col not in df.columns:
            print(f"  WARNING: Column '{model_col}' not found in data, skipping")
            continue

        sneed_col = f"s_need_{model_col}"
        if sneed_col not in df.columns:
            df[sneed_col] = np.nan

        # Find rows that still need scoring (null s_need)
        missing_mask = df[sneed_col].isna()
        for idx in df.index[missing_mask]:
            response = df.at[idx, model_col]
            if pd.isna(response) or str(response).strip() == "":
                continue
            pairs_to_score.append({
                "idx": idx,
                "model_col": model_col,
                "prompt": df.at[idx, prompt_col],
                "response": response,
            })

    if not pairs_to_score:
        print("  All pairs already scored, skipping Stage 2")
        return df

    print(f"  Scoring {len(pairs_to_score)} (prompt, model) pairs...")

    # Score in batches with progress tracking
    async def score_one(pair):
        result = await score_with_judge_async(
            client,
            pair["prompt"],
            divergence_type,
            pair["response"],
            judge_template,
            semaphore,
            model=judge_cfg["model"],
            temperature=judge_cfg["temperature"],
        )
        return pair["idx"], pair["model_col"], result

    tasks = [score_one(p) for p in pairs_to_score]
    results = await async_tqdm.gather(*tasks, desc="S_need scoring")

    # Populate dataframe
    n_errors = 0
    for idx, model_col, result in results:
        sneed_col = f"s_need_{model_col}"
        nf_col = f"nf_{model_col}"
        ea_col = f"ea_{model_col}"
        ar_col = f"ar_{model_col}"
        err_col = f"err_{model_col}"

        # Ensure sub-score columns exist
        for col in [nf_col, ea_col, ar_col, err_col]:
            if col not in df.columns:
                df[col] = np.nan if not col.startswith("err_") else None

        if result["error"]:
            n_errors += 1
            df.at[idx, err_col] = result["error"]
        else:
            df.at[idx, sneed_col] = result["s_need"]
            df.at[idx, nf_col] = result["need_fulfillment"]
            df.at[idx, ea_col] = result["expert_approval"]
            df.at[idx, ar_col] = result["appropriate_redirection"]

    if n_errors > 0:
        print(f"  WARNING: {n_errors} scoring errors")

    return df


# ─── Stage 3: R_pref Merge ──────────────────────────────────────────────────

def merge_rpref(df, config):
    """Merge R_pref scores from companion notebook if available."""
    rpref_path = Path(__file__).parent / "data" / "cross_model_rpref.parquet"
    if not rpref_path.exists():
        print("  R_pref file not found — R_pref columns will show 'TBD' in Table 4")
        return df, False

    rpref_df = pd.read_parquet(rpref_path)
    print(f"  Loaded R_pref scores: {len(rpref_df)} rows, {len(rpref_df.columns)} columns")

    # Merge on prompt_id
    df = df.merge(rpref_df, on="prompt_id", how="left", suffixes=("", "_rpref"))
    print(f"  Merged R_pref on prompt_id")
    return df, True


# ─── Stage 4: Analysis & Output ─────────────────────────────────────────────

def generate_table_4(df, config, has_rpref):
    """Generate Table 4: Cross-model S_need statistics."""
    cfg = config["step_6"]
    model_columns = cfg["model_columns"]
    human_label_col = cfg["human_label_col"]

    # Filter to models actually present in the data
    available_models = [m for m in model_columns if f"s_need_{m}" in df.columns]

    # Collect human labels (binary)
    y_human = df[human_label_col].values.astype(int)

    # Global median threshold for binarization
    all_sneed = []
    for model_col in available_models:
        vals = df[f"s_need_{model_col}"].dropna().values
        all_sneed.extend(vals)
    global_median = np.median(all_sneed) if all_sneed else 0.5

    # Per-model stats
    table_rows = []
    all_raw_p = []
    p_model_map = []  # (row_index, model_name)

    for model_col in available_models:
        sneed_col = f"s_need_{model_col}"
        valid_mask = df[sneed_col].notna()
        valid_df = df[valid_mask]
        scores = valid_df[sneed_col].values.astype(float)
        n = len(scores)

        if n == 0:
            continue

        # Mean and bootstrap CI (each prompt is its own cluster here)
        mean_sneed = float(scores.mean())
        clusters = valid_df["prompt_id"].values
        ci_lower, ci_upper = clustered_bootstrap_ci(scores, clusters)

        # Wilcoxon signed-rank one-sample vs 1.0
        diff_from_ideal = scores - 1.0
        if np.all(diff_from_ideal == 0):
            w_p = 1.0
        else:
            _, w_p = wilcoxon(diff_from_ideal)

        # Cohen's d_z vs 1.0
        ideal = np.ones_like(scores)
        d_z = cohens_d_paired(scores, ideal)

        # Human agreement: binarize S_need at global median
        sneed_binary = (scores >= global_median).astype(int)
        human_labels = valid_df[human_label_col].values.astype(int)
        agree_pct = float((sneed_binary == human_labels).mean() * 100)

        # ROC-AUC vs human
        roc_auc = compute_roc_auc(human_labels, scores)

        row = {
            "Model": model_col,
            "n": n,
            "Mean S_need": round(mean_sneed, 4),
            "S_need CI": f"({ci_lower:.4f}, {ci_upper:.4f})",
            "S_need p": w_p,
            "S_need d": round(d_z, 4),
            "Human Agree %": round(agree_pct, 1),
            "ROC-AUC": round(roc_auc, 4),
        }

        # R_pref columns
        rpref_col = f"rpref_mean_{model_col}"
        if has_rpref and rpref_col in df.columns:
            rpref_vals = valid_df[rpref_col].dropna().values
            if len(rpref_vals) > 0:
                rpref_mean = float(rpref_vals.mean())
                rpref_clusters = valid_df.loc[valid_df[rpref_col].notna(), "prompt_id"].values
                rpref_ci = clustered_bootstrap_ci(rpref_vals, rpref_clusters)
                row["Mean R_pref"] = round(rpref_mean, 4)
                row["R_pref CI"] = f"({rpref_ci[0]:.4f}, {rpref_ci[1]:.4f})"
            else:
                row["Mean R_pref"] = "TBD"
                row["R_pref CI"] = "TBD"
        else:
            row["Mean R_pref"] = "TBD"
            row["R_pref CI"] = "TBD"

        all_raw_p.append(w_p)
        p_model_map.append(len(table_rows))
        table_rows.append(row)

    # BH correction across all per-model p-values
    if all_raw_p:
        adjusted_p = benjamini_hochberg(all_raw_p)
        for i, row_idx in enumerate(p_model_map):
            table_rows[row_idx]["S_need p"] = float(adjusted_p[i])

    table_df = pd.DataFrame(table_rows)
    return table_df, global_median


def generate_analysis_json(df, config, has_rpref, global_median):
    """Generate full statistical analysis results."""
    cfg = config["step_6"]
    model_columns = cfg["model_columns"]
    human_label_col = cfg["human_label_col"]

    available_models = [m for m in model_columns if f"s_need_{m}" in df.columns]

    # ── Per-model results ──
    per_model = {}
    all_per_model_p = []
    per_model_names = []

    for model_col in available_models:
        sneed_col = f"s_need_{model_col}"
        valid_mask = df[sneed_col].notna()
        valid_df = df[valid_mask]
        scores = valid_df[sneed_col].values.astype(float)
        n = len(scores)

        if n == 0:
            continue

        mean_sneed = float(scores.mean())
        clusters = valid_df["prompt_id"].values
        ci_lower, ci_upper = clustered_bootstrap_ci(scores, clusters)

        diff_from_ideal = scores - 1.0
        if np.all(diff_from_ideal == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = wilcoxon(diff_from_ideal)

        ideal = np.ones_like(scores)
        d_z = cohens_d_paired(scores, ideal)

        human_labels = valid_df[human_label_col].values.astype(int)
        sneed_binary = (scores >= global_median).astype(int)
        agree_pct = float((sneed_binary == human_labels).mean() * 100)
        roc_auc = compute_roc_auc(human_labels, scores)

        # Point-biserial correlation
        pb_r, pb_p = pointbiserialr(human_labels, scores)

        model_result = {
            "n": n,
            "mean_s_need": mean_sneed,
            "std_s_need": float(scores.std()),
            "ci_95": [ci_lower, ci_upper],
            "wilcoxon_vs_1": {"statistic": float(w_stat), "p_raw": float(w_p)},
            "cohens_d_z_vs_1": float(d_z),
            "human_agree_pct": agree_pct,
            "roc_auc_vs_human": float(roc_auc),
            "point_biserial_vs_human": {"r": float(pb_r), "p": float(pb_p)},
            "global_median_threshold": float(global_median),
        }

        # S_need vs R_pref correlation (if available)
        rpref_col = f"rpref_mean_{model_col}"
        if has_rpref and rpref_col in df.columns:
            both_valid = valid_df[rpref_col].notna()
            if both_valid.sum() >= 3:
                s_vals = valid_df.loc[both_valid, sneed_col].values.astype(float)
                r_vals = valid_df.loc[both_valid, rpref_col].values.astype(float)
                rho, rho_p = spearmanr(s_vals, r_vals)
                model_result["spearman_sneed_rpref"] = {
                    "rho": float(rho),
                    "p": float(rho_p),
                    "n": int(both_valid.sum()),
                }

        all_per_model_p.append(w_p)
        per_model_names.append(model_col)
        per_model[model_col] = model_result

    # BH correction for per-model tests
    if all_per_model_p:
        adjusted = benjamini_hochberg(all_per_model_p)
        for i, name in enumerate(per_model_names):
            per_model[name]["wilcoxon_vs_1"]["p_bh"] = float(adjusted[i])

    # ── Pairwise tests ──
    pairwise = []
    all_pairwise_p = []

    model_pairs = list(combinations(available_models, 2))
    for model_a, model_b in model_pairs:
        col_a = f"s_need_{model_a}"
        col_b = f"s_need_{model_b}"

        # Both must have valid scores on the same prompts
        both_valid = df[col_a].notna() & df[col_b].notna()
        paired_df = df[both_valid]
        n_paired = len(paired_df)

        if n_paired < 3:
            continue

        scores_a = paired_df[col_a].values.astype(float)
        scores_b = paired_df[col_b].values.astype(float)

        diff = scores_a - scores_b
        if np.all(diff == 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_stat, w_p = wilcoxon(diff)

        d_z = cohens_d_paired(scores_a, scores_b)

        all_pairwise_p.append(w_p)
        pairwise.append({
            "model_a": model_a,
            "model_b": model_b,
            "n_paired": n_paired,
            "mean_a": float(scores_a.mean()),
            "mean_b": float(scores_b.mean()),
            "mean_diff": float(diff.mean()),
            "wilcoxon": {"statistic": float(w_stat), "p_raw": float(w_p)},
            "cohens_d_z": float(d_z),
        })

    # BH correction for pairwise tests
    if all_pairwise_p:
        adjusted = benjamini_hochberg(all_pairwise_p)
        for i, pair in enumerate(pairwise):
            pair["wilcoxon"]["p_bh"] = float(adjusted[i])

    # ── Assemble output ──
    analysis = {
        "config": {
            "sample_size": cfg["sample_size"],
            "random_seed": cfg["random_seed"],
            "divergence_type": cfg["divergence_type"],
            "judge_model": cfg["judge"]["model"],
            "n_models": len(available_models),
            "models": available_models,
            "global_median_threshold": float(global_median),
        },
        "per_model": per_model,
        "pairwise": pairwise,
        "statistical_notes": {
            "per_model_p": "Wilcoxon signed-rank one-sample vs 1.0, BH-FDR corrected",
            "per_model_d": "Cohen's d_z (paired) vs ideal score of 1.0",
            "per_model_ci": "95% clustered bootstrap (10000 resamples, clustered by prompt_id)",
            "pairwise_p": "Wilcoxon signed-rank paired by prompt, BH-FDR corrected",
            "pairwise_d": "Cohen's d_z for paired difference between models",
            "human_agree": f"Binarized S_need at global median ({global_median:.4f}) vs validation_human",
            "roc_auc": "Continuous S_need vs binary validation_human",
        },
    }

    return analysis


# ─── Main orchestrator ───────────────────────────────────────────────────────

async def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        exit(1)

    config = load_config()
    judge_template = load_judge_prompt(config)
    cfg = config["step_6"]
    output_cfg = cfg["output"]

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("Step 6: Cross-Model Need Gap Analysis")
    print("=" * 80)

    # ── Stage 1: Load & Sample ──
    sampled_path = base_dir / output_cfg["sampled_parquet"]
    if sampled_path.exists():
        print(f"\n[Stage 1] Loading existing sample from {sampled_path}")
        df = pd.read_parquet(sampled_path)
        print(f"  Loaded {len(df)} rows")
    else:
        print(f"\n[Stage 1] Loading bench data and stratified sampling...")
        bench_df = load_bench_data(config)
        df = stratified_sample(bench_df, config)
        sampled_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(sampled_path, index=False)
        print(f"  Saved sample to {sampled_path}")

    # ── Stage 2: S_need Scoring ──
    scored_path = base_dir / output_cfg["scored_parquet"]
    if scored_path.exists():
        print(f"\n[Stage 2] Loading existing scores from {scored_path}")
        df = pd.read_parquet(scored_path)
        # Check for missing scores and resume if needed
        model_columns = cfg["model_columns"]
        missing_count = 0
        for model_col in model_columns:
            sneed_col = f"s_need_{model_col}"
            if sneed_col in df.columns:
                missing_count += df[sneed_col].isna().sum()
            else:
                # Column doesn't exist at all — count all rows
                missing_count += len(df)

        if missing_count > 0:
            print(f"  Found {missing_count} missing scores, resuming...")
            client = AsyncOpenAI(api_key=api_key)
            df = await score_all_models_sneed(client, df, config, judge_template)
            df.to_parquet(scored_path, index=False)
            print(f"  Saved updated scores to {scored_path}")
        else:
            print(f"  All scores present ({len(df)} rows)")
    else:
        print(f"\n[Stage 2] Scoring all (prompt, model) pairs...")
        client = AsyncOpenAI(api_key=api_key)
        df = await score_all_models_sneed(client, df, config, judge_template)
        scored_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(scored_path, index=False)
        print(f"  Saved scores to {scored_path}")

    # ── Stage 3: R_pref Merge ──
    print(f"\n[Stage 3] Checking for R_pref data...")
    df, has_rpref = merge_rpref(df, config)

    # ── Stage 4: Analysis & Output ──
    print(f"\n[Stage 4] Computing statistics and generating outputs...")

    table_4_df, global_median = generate_table_4(df, config, has_rpref)
    analysis = generate_analysis_json(df, config, has_rpref, global_median)

    # Save Table 4
    table_path = base_dir / output_cfg["table_csv"]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_4_df.to_csv(table_path, index=False)
    print(f"\n  Table 4 saved to: {table_path}")

    # Save analysis JSON
    json_path = base_dir / output_cfg["analysis_json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Analysis saved to: {json_path}")

    # ── Print summary ──
    print(f"\n{'='*80}")
    print("Table 4: Cross-Model S_need Comparison")
    print(f"{'='*80}")
    print(table_4_df.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"Pairwise comparisons: {len(analysis['pairwise'])} pairs tested")
    sig_pairs = [p for p in analysis["pairwise"] if p["wilcoxon"].get("p_bh", 1.0) < 0.05]
    print(f"Significant (BH-corrected p < 0.05): {len(sig_pairs)} pairs")

    if sig_pairs:
        print("\nSignificant pairwise differences:")
        for p in sorted(sig_pairs, key=lambda x: x["wilcoxon"]["p_bh"]):
            print(f"  {p['model_a']} vs {p['model_b']}: "
                  f"diff={p['mean_diff']:+.4f}, d={p['cohens_d_z']:.3f}, "
                  f"p_bh={p['wilcoxon']['p_bh']:.4e}")

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
