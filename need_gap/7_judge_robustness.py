"""
Step 7: Judge Robustness — Multi-Judge Agreement & Reversed Control.

Two experiments to address the self-preference bias critique:

1. Multi-Judge Agreement: Re-score all responses with GPT-4o-mini. If a different
   model produces the same S_need rankings, the signal is in the response content,
   not model self-recognition.

2. Reversed Control: Have GPT-4o generate literal/preference-style responses (just
   answer the query, no need-awareness). Score these with the S_need judge. If GPT-4o
   gives its own literal responses low S_need, the judge is content-sensitive, not
   origin-sensitive.

Stages (with checkpoint/resume):
  1. Generate literal responses with GPT-4o
  2. Score all response types with GPT-4o-mini
  3. Score literal responses with GPT-4o
  4. Compute agreement metrics & output
"""

import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from scipy.stats import wilcoxon, spearmanr, pearsonr
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


def load_literal_prompt(config):
    prompt_path = Path(__file__).parent / config["step_7"]["literal_generator"]["prompt_path"]
    with open(prompt_path, "r") as f:
        return f.read()


# ─── Statistical utilities (from 5_analyze_results.py) ──────────────────────

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


# ─── Async scoring (parameterized model, from step 6 pattern) ───────────────

async def score_with_judge_async(client, prompt, divergence_type, response,
                                  judge_template, semaphore,
                                  model="gpt-4o", temperature=0.3):
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


# ─── Async literal response generation ──────────────────────────────────────

async def generate_literal_response_async(client, prompt_text, template,
                                           semaphore, model="gpt-4o",
                                           temperature=0.7, max_tokens=1000):
    """Generate a literal/preference-style response (no need-awareness)."""
    filled_prompt = template.replace("{prompt}", str(prompt_text))

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": filled_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return {
            "response": response.choices[0].message.content,
            "error": None,
        }
    except Exception as e:
        return {
            "response": None,
            "error": str(e),
        }


# ─── Stage 1: Generate literal responses ────────────────────────────────────

async def stage_1_generate_literal(client, df, config):
    """Generate literal responses with GPT-4o for all prompts."""
    cfg = config["step_7"]
    gen_cfg = cfg["literal_generator"]
    semaphore = asyncio.Semaphore(cfg["max_concurrent"])

    template = load_literal_prompt(config)

    # Check for resume: skip rows that already have literal_response
    if "literal_response" in df.columns:
        missing_mask = df["literal_response"].isna() | (df["literal_response"] == "")
    else:
        df["literal_response"] = None
        missing_mask = pd.Series(True, index=df.index)

    indices_to_generate = df.index[missing_mask].tolist()

    if len(indices_to_generate) == 0:
        print("  All literal responses already generated, skipping Stage 1")
        return df

    print(f"  Generating {len(indices_to_generate)} literal responses...")

    async def gen_one(idx):
        result = await generate_literal_response_async(
            client,
            df.at[idx, "test_prompt"],
            template,
            semaphore,
            model=gen_cfg["model"],
            temperature=gen_cfg["temperature"],
            max_tokens=gen_cfg["max_tokens"],
        )
        return idx, result

    tasks = [gen_one(idx) for idx in indices_to_generate]
    results = await async_tqdm.gather(*tasks, desc="Literal responses")

    n_errors = 0
    for idx, result in results:
        if result["error"]:
            n_errors += 1
            print(f"  Error at index {idx}: {result['error']}")
        else:
            df.at[idx, "literal_response"] = result["response"]

    if n_errors > 0:
        print(f"  WARNING: {n_errors} generation errors")

    return df


# ─── Stage 2: Score all response types with GPT-4o-mini ─────────────────────

async def stage_2_score_with_mini(client, df, config, judge_template):
    """Score preference-matched, need-aware, and literal responses with GPT-4o-mini."""
    cfg = config["step_7"]
    judge_cfg = cfg["judges"][0]  # gpt-4o-mini
    semaphore = asyncio.Semaphore(cfg["max_concurrent"])

    response_types = [
        ("preference_matched_response", "mini_pref"),
        ("need_aware_response", "mini_need"),
        ("literal_response", "mini_literal"),
    ]

    for response_col, prefix in response_types:
        sneed_col = f"{prefix}_s_need"
        nf_col = f"{prefix}_nf"
        ea_col = f"{prefix}_ea"
        ar_col = f"{prefix}_ar"

        # Ensure columns exist
        for col in [sneed_col, nf_col, ea_col, ar_col]:
            if col not in df.columns:
                df[col] = np.nan

        # Find rows that need scoring
        missing_mask = df[sneed_col].isna()
        # Also skip rows where the response itself is missing
        if response_col in df.columns:
            valid_response = df[response_col].notna() & (df[response_col] != "")
            missing_mask = missing_mask & valid_response

        indices_to_score = df.index[missing_mask].tolist()

        if len(indices_to_score) == 0:
            print(f"  [{prefix}] All scores present, skipping")
            continue

        print(f"  [{prefix}] Scoring {len(indices_to_score)} responses with {judge_cfg['model']}...")

        async def score_one(idx, resp_col=response_col):
            result = await score_with_judge_async(
                client,
                df.at[idx, "test_prompt"],
                df.at[idx, "divergence_type"],
                df.at[idx, resp_col],
                judge_template,
                semaphore,
                model=judge_cfg["model"],
                temperature=judge_cfg["temperature"],
            )
            return idx, result

        tasks = [score_one(idx) for idx in indices_to_score]
        results = await async_tqdm.gather(*tasks, desc=f"{prefix} scoring")

        n_errors = 0
        for idx, result in results:
            if result["error"]:
                n_errors += 1
            else:
                df.at[idx, sneed_col] = result["s_need"]
                df.at[idx, nf_col] = result["need_fulfillment"]
                df.at[idx, ea_col] = result["expert_approval"]
                df.at[idx, ar_col] = result["appropriate_redirection"]

        if n_errors > 0:
            print(f"  [{prefix}] WARNING: {n_errors} scoring errors")

    return df


# ─── Stage 3: Score literal responses with GPT-4o ───────────────────────────

async def stage_3_score_literal_gpt4o(client, df, config, judge_template):
    """Score literal responses with the original GPT-4o judge."""
    cfg = config["step_7"]
    semaphore = asyncio.Semaphore(cfg["max_concurrent"])

    prefix = "literal"
    sneed_col = f"{prefix}_s_need"
    nf_col = f"{prefix}_nf"
    ea_col = f"{prefix}_ea"
    ar_col = f"{prefix}_ar"

    for col in [sneed_col, nf_col, ea_col, ar_col]:
        if col not in df.columns:
            df[col] = np.nan

    # Find rows needing scoring
    missing_mask = df[sneed_col].isna()
    valid_response = df["literal_response"].notna() & (df["literal_response"] != "")
    missing_mask = missing_mask & valid_response

    indices_to_score = df.index[missing_mask].tolist()

    if len(indices_to_score) == 0:
        print("  All literal GPT-4o scores present, skipping Stage 3")
        return df

    print(f"  Scoring {len(indices_to_score)} literal responses with GPT-4o...")

    async def score_one(idx):
        result = await score_with_judge_async(
            client,
            df.at[idx, "test_prompt"],
            df.at[idx, "divergence_type"],
            df.at[idx, "literal_response"],
            judge_template,
            semaphore,
            model="gpt-4o",
            temperature=0.3,
        )
        return idx, result

    tasks = [score_one(idx) for idx in indices_to_score]
    results = await async_tqdm.gather(*tasks, desc="Literal GPT-4o scoring")

    n_errors = 0
    for idx, result in results:
        if result["error"]:
            n_errors += 1
        else:
            df.at[idx, sneed_col] = result["s_need"]
            df.at[idx, nf_col] = result["need_fulfillment"]
            df.at[idx, ea_col] = result["expert_approval"]
            df.at[idx, ar_col] = result["appropriate_redirection"]

    if n_errors > 0:
        print(f"  WARNING: {n_errors} scoring errors")

    return df


# ─── Stage 4: Compute agreement metrics & output ────────────────────────────

def compute_cohens_kappa(y1, y2):
    """Compute Cohen's kappa for two binary arrays."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return float("nan")

    # Observed agreement
    p_o = (y1 == y2).sum() / n

    # Expected agreement
    p1_pos = y1.sum() / n
    p2_pos = y2.sum() / n
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    return (p_o - p_e) / (1 - p_e)


def stage_4_compute_metrics(df):
    """Compute all agreement and reversed control metrics."""
    results = {}

    # ── Filter to rows with complete data ──
    required_cols = [
        "pref_s_need", "need_s_need",
        "mini_pref_s_need", "mini_need_s_need", "mini_literal_s_need",
        "literal_s_need",
    ]
    complete_mask = pd.Series(True, index=df.index)
    for col in required_cols:
        if col in df.columns:
            complete_mask = complete_mask & df[col].notna()
        else:
            print(f"  WARNING: Missing column {col}")
            complete_mask = pd.Series(False, index=df.index)

    df_complete = df[complete_mask].copy()
    n_complete = len(df_complete)
    print(f"  Complete data for {n_complete}/{len(df)} rows")

    if n_complete == 0:
        print("  ERROR: No complete rows found!")
        return {}, pd.DataFrame()

    # ── Sub-score columns ──
    sub_criteria = ["nf", "ea", "ar"]
    criteria_names = {
        "nf": "need_fulfillment",
        "ea": "expert_approval",
        "ar": "appropriate_redirection",
    }

    # ══════════════════════════════════════════════════════════════════════
    # MULTI-JUDGE AGREEMENT (GPT-4o vs GPT-4o-mini)
    # ══════════════════════════════════════════════════════════════════════

    agreement = {}

    # Build paired arrays: (gpt4o_score, mini_score) for all response types
    gpt4o_scores = np.concatenate([
        df_complete["pref_s_need"].values,
        df_complete["need_s_need"].values,
        df_complete["literal_s_need"].values,
    ])
    mini_scores = np.concatenate([
        df_complete["mini_pref_s_need"].values,
        df_complete["mini_need_s_need"].values,
        df_complete["mini_literal_s_need"].values,
    ])

    # Pooled correlations
    spearman_rho, spearman_p = spearmanr(gpt4o_scores, mini_scores)
    pearson_r, pearson_p = pearsonr(gpt4o_scores, mini_scores)

    agreement["pooled"] = {
        "n_pairs": len(gpt4o_scores),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "mean_abs_diff": float(np.abs(gpt4o_scores - mini_scores).mean()),
        "std_abs_diff": float(np.abs(gpt4o_scores - mini_scores).std()),
    }

    # Cohen's kappa at two thresholds
    median_threshold = float(np.median(gpt4o_scores))
    for threshold_name, threshold_val in [("median", median_threshold), ("0.8", 0.8)]:
        gpt4o_binary = (gpt4o_scores >= threshold_val).astype(int)
        mini_binary = (mini_scores >= threshold_val).astype(int)
        kappa = compute_cohens_kappa(gpt4o_binary, mini_binary)
        agreement["pooled"][f"cohens_kappa_threshold_{threshold_name}"] = float(kappa)

    agreement["pooled"]["median_threshold_value"] = median_threshold

    # Per-response-type correlations
    response_type_pairs = [
        ("pref", "pref_s_need", "mini_pref_s_need"),
        ("need", "need_s_need", "mini_need_s_need"),
        ("literal", "literal_s_need", "mini_literal_s_need"),
    ]

    for rtype, gpt4o_col, mini_col in response_type_pairs:
        g = df_complete[gpt4o_col].values
        m = df_complete[mini_col].values
        rho, rho_p = spearmanr(g, m)
        r, r_p = pearsonr(g, m)
        agreement[rtype] = {
            "n": len(g),
            "gpt4o_mean": float(g.mean()),
            "mini_mean": float(m.mean()),
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
            "pearson_r": float(r),
            "pearson_p": float(r_p),
            "mean_abs_diff": float(np.abs(g - m).mean()),
        }

    # Per-criterion correlations (pooled across response types)
    # Existing data uses: pref_need_fulfillment, need_need_fulfillment
    # New columns use: mini_pref_nf, mini_need_nf, literal_nf, mini_literal_nf
    per_criterion_agreement = {}
    for sc, cname in criteria_names.items():
        # GPT-4o sub-scores: existing pref/need use full names, literal uses abbreviation
        gpt4o_parts = []
        mini_parts = []

        # Preference-matched
        gpt4o_col = f"pref_{cname}"
        mini_col = f"mini_pref_{sc}"
        if gpt4o_col in df_complete.columns and mini_col in df_complete.columns:
            gpt4o_parts.append(df_complete[gpt4o_col].values)
            mini_parts.append(df_complete[mini_col].values)

        # Need-aware
        gpt4o_col = f"need_{cname}"
        mini_col = f"mini_need_{sc}"
        if gpt4o_col in df_complete.columns and mini_col in df_complete.columns:
            gpt4o_parts.append(df_complete[gpt4o_col].values)
            mini_parts.append(df_complete[mini_col].values)

        # Literal
        gpt4o_col = f"literal_{sc}"
        mini_col = f"mini_literal_{sc}"
        if gpt4o_col in df_complete.columns and mini_col in df_complete.columns:
            gpt4o_parts.append(df_complete[gpt4o_col].values)
            mini_parts.append(df_complete[mini_col].values)

        if gpt4o_parts and mini_parts:
            gpt4o_sub = np.concatenate(gpt4o_parts)
            mini_sub = np.concatenate(mini_parts)

            if len(gpt4o_sub) > 2 and len(gpt4o_sub) == len(mini_sub):
                valid = ~(np.isnan(gpt4o_sub) | np.isnan(mini_sub))
                if valid.sum() > 2:
                    rho, rho_p = spearmanr(gpt4o_sub[valid], mini_sub[valid])
                    per_criterion_agreement[cname] = {
                        "n": int(valid.sum()),
                        "spearman_rho": float(rho),
                        "spearman_p": float(rho_p),
                    }

    agreement["per_criterion"] = per_criterion_agreement

    # Critical test: Does GPT-4o-mini reproduce the S_need gap pattern?
    divergent_mask = df_complete["divergence_type"] != "none"
    control_mask = df_complete["divergence_type"] == "none"

    gap_reproduction = {}
    for subset_name, mask in [("divergent", divergent_mask), ("control", control_mask)]:
        sub = df_complete[mask]
        if len(sub) == 0:
            continue

        # GPT-4o gaps
        gpt4o_gap = sub["need_s_need"].values - sub["pref_s_need"].values
        # GPT-4o-mini gaps
        mini_gap = sub["mini_need_s_need"].values - sub["mini_pref_s_need"].values

        clusters = sub["person_id"].values

        gap_reproduction[subset_name] = {
            "n": len(sub),
            "gpt4o_gap_mean": float(gpt4o_gap.mean()),
            "gpt4o_gap_ci": list(clustered_bootstrap_ci(gpt4o_gap, clusters)),
            "mini_gap_mean": float(mini_gap.mean()),
            "mini_gap_ci": list(clustered_bootstrap_ci(mini_gap, clusters)),
        }

        # Correlation of per-prompt gaps between judges
        if len(gpt4o_gap) > 2:
            rho, rho_p = spearmanr(gpt4o_gap, mini_gap)
            gap_reproduction[subset_name]["gap_spearman_rho"] = float(rho)
            gap_reproduction[subset_name]["gap_spearman_p"] = float(rho_p)

    agreement["gap_reproduction"] = gap_reproduction

    results["multi_judge_agreement"] = agreement

    # ══════════════════════════════════════════════════════════════════════
    # REVERSED CONTROL (Content-sensitivity test)
    # ══════════════════════════════════════════════════════════════════════

    reversed_control = {}

    # Collect all raw p-values for BH correction
    all_p_values = []
    p_value_labels = []

    # Comparison 1: literal vs pref-matched (GPT-4o judge)
    literal_scores = df_complete["literal_s_need"].values
    pref_scores = df_complete["pref_s_need"].values
    need_scores = df_complete["need_s_need"].values
    clusters = df_complete["person_id"].values

    # literal vs pref
    diff_lit_pref = literal_scores - pref_scores
    if not np.all(diff_lit_pref == 0):
        _, p_lit_pref = wilcoxon(diff_lit_pref)
    else:
        p_lit_pref = 1.0
    d_lit_pref = cohens_d_paired(literal_scores, pref_scores)
    all_p_values.append(p_lit_pref)
    p_value_labels.append("literal_vs_pref_overall")

    reversed_control["literal_vs_pref_overall"] = {
        "n": n_complete,
        "literal_mean": float(literal_scores.mean()),
        "literal_ci": list(clustered_bootstrap_ci(literal_scores, clusters)),
        "pref_mean": float(pref_scores.mean()),
        "pref_ci": list(clustered_bootstrap_ci(pref_scores, clusters)),
        "mean_diff": float(diff_lit_pref.mean()),
        "diff_ci": list(clustered_bootstrap_ci(diff_lit_pref, clusters)),
        "wilcoxon_p_raw": float(p_lit_pref),
        "cohens_d": float(d_lit_pref),
    }

    # literal vs need-aware
    diff_lit_need = literal_scores - need_scores
    if not np.all(diff_lit_need == 0):
        _, p_lit_need = wilcoxon(diff_lit_need)
    else:
        p_lit_need = 1.0
    d_lit_need = cohens_d_paired(literal_scores, need_scores)
    all_p_values.append(p_lit_need)
    p_value_labels.append("literal_vs_need_overall")

    reversed_control["literal_vs_need_overall"] = {
        "n": n_complete,
        "literal_mean": float(literal_scores.mean()),
        "need_mean": float(need_scores.mean()),
        "mean_diff": float(diff_lit_need.mean()),
        "diff_ci": list(clustered_bootstrap_ci(diff_lit_need, clusters)),
        "wilcoxon_p_raw": float(p_lit_need),
        "cohens_d": float(d_lit_need),
    }

    # Split by divergent vs control
    for subset_name, mask in [("divergent", divergent_mask), ("control", control_mask)]:
        sub = df_complete[mask]
        if len(sub) == 0:
            continue

        s_lit = sub["literal_s_need"].values
        s_pref = sub["pref_s_need"].values
        s_need = sub["need_s_need"].values
        s_clusters = sub["person_id"].values

        # literal vs pref
        d1 = s_lit - s_pref
        if not np.all(d1 == 0):
            _, p1 = wilcoxon(d1)
        else:
            p1 = 1.0
        d1_d = cohens_d_paired(s_lit, s_pref)
        all_p_values.append(p1)
        p_value_labels.append(f"literal_vs_pref_{subset_name}")

        # literal vs need
        d2 = s_lit - s_need
        if not np.all(d2 == 0):
            _, p2 = wilcoxon(d2)
        else:
            p2 = 1.0
        d2_d = cohens_d_paired(s_lit, s_need)
        all_p_values.append(p2)
        p_value_labels.append(f"literal_vs_need_{subset_name}")

        reversed_control[subset_name] = {
            "n": len(sub),
            "literal_mean": float(s_lit.mean()),
            "literal_ci": list(clustered_bootstrap_ci(s_lit, s_clusters)),
            "pref_mean": float(s_pref.mean()),
            "pref_ci": list(clustered_bootstrap_ci(s_pref, s_clusters)),
            "need_mean": float(s_need.mean()),
            "need_ci": list(clustered_bootstrap_ci(s_need, s_clusters)),
            "literal_vs_pref": {
                "mean_diff": float(d1.mean()),
                "diff_ci": list(clustered_bootstrap_ci(d1, s_clusters)),
                "wilcoxon_p_raw": float(p1),
                "cohens_d": float(d1_d),
            },
            "literal_vs_need": {
                "mean_diff": float(d2.mean()),
                "diff_ci": list(clustered_bootstrap_ci(d2, s_clusters)),
                "wilcoxon_p_raw": float(p2),
                "cohens_d": float(d2_d),
            },
        }

    # BH-FDR correction across all reversed control p-values
    adjusted_p = benjamini_hochberg(all_p_values)
    p_adjusted_map = dict(zip(p_value_labels, adjusted_p))

    # Patch adjusted p-values into results
    reversed_control["literal_vs_pref_overall"]["wilcoxon_p_bh"] = float(
        p_adjusted_map["literal_vs_pref_overall"]
    )
    reversed_control["literal_vs_need_overall"]["wilcoxon_p_bh"] = float(
        p_adjusted_map["literal_vs_need_overall"]
    )
    for subset_name in ["divergent", "control"]:
        if subset_name in reversed_control:
            key1 = f"literal_vs_pref_{subset_name}"
            key2 = f"literal_vs_need_{subset_name}"
            if key1 in p_adjusted_map:
                reversed_control[subset_name]["literal_vs_pref"]["wilcoxon_p_bh"] = float(
                    p_adjusted_map[key1]
                )
            if key2 in p_adjusted_map:
                reversed_control[subset_name]["literal_vs_need"]["wilcoxon_p_bh"] = float(
                    p_adjusted_map[key2]
                )

    results["reversed_control"] = reversed_control

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY VERDICTS
    # ══════════════════════════════════════════════════════════════════════

    summary = {}

    # Multi-judge agreement verdict
    pooled_rho = agreement["pooled"]["spearman_rho"]
    summary["multi_judge_spearman_rho"] = pooled_rho
    summary["multi_judge_agreement_strong"] = pooled_rho > 0.7
    summary["multi_judge_agreement_moderate"] = pooled_rho > 0.5

    # Gap reproduction verdict
    if "divergent" in agreement["gap_reproduction"]:
        div_gap = agreement["gap_reproduction"]["divergent"]
        summary["mini_reproduces_divergent_gap"] = div_gap["mini_gap_mean"] > 0
        summary["divergent_gap_gpt4o"] = div_gap["gpt4o_gap_mean"]
        summary["divergent_gap_mini"] = div_gap["mini_gap_mean"]

    if "control" in agreement["gap_reproduction"]:
        ctrl_gap = agreement["gap_reproduction"]["control"]
        summary["control_gap_gpt4o"] = ctrl_gap["gpt4o_gap_mean"]
        summary["control_gap_mini"] = ctrl_gap["mini_gap_mean"]

    # Reversed control verdict
    if "divergent" in reversed_control:
        div_rc = reversed_control["divergent"]
        lit_pref_d = abs(div_rc["literal_vs_pref"]["cohens_d"])
        lit_need_d = abs(div_rc["literal_vs_need"]["cohens_d"])
        summary["literal_similar_to_pref"] = lit_pref_d < 0.3
        summary["literal_different_from_need"] = lit_need_d > 0.3
        summary["content_sensitive_not_origin_sensitive"] = (
            lit_pref_d < 0.3 and lit_need_d > 0.3
        )
        summary["literal_vs_pref_d_divergent"] = float(div_rc["literal_vs_pref"]["cohens_d"])
        summary["literal_vs_need_d_divergent"] = float(div_rc["literal_vs_need"]["cohens_d"])

    results["summary"] = summary

    # ══════════════════════════════════════════════════════════════════════
    # TABLE 5: Paper-ready summary
    # ══════════════════════════════════════════════════════════════════════

    table_rows = []

    # Multi-judge agreement section
    for rtype, gpt4o_col, mini_col in response_type_pairs:
        rtype_label = {
            "pref": "Preference-matched",
            "need": "Need-aware",
            "literal": "Literal",
        }[rtype]

        g = df_complete[gpt4o_col].values
        m = df_complete[mini_col].values
        rho, _ = spearmanr(g, m)

        table_rows.append({
            "Experiment": "Multi-Judge Agreement",
            "Comparison": f"{rtype_label} (GPT-4o vs Mini)",
            "GPT-4o Mean": round(float(g.mean()), 4),
            "GPT-4o-mini Mean": round(float(m.mean()), 4),
            "Spearman rho": round(float(rho), 4),
            "MAD": round(float(np.abs(g - m).mean()), 4),
            "Cohen's d": "",
            "p (BH)": "",
        })

    # Gap reproduction rows
    for subset_name in ["divergent", "control"]:
        if subset_name not in agreement["gap_reproduction"]:
            continue
        gap = agreement["gap_reproduction"][subset_name]
        label = subset_name.title()
        table_rows.append({
            "Experiment": "Gap Reproduction",
            "Comparison": f"{label} S_need gap",
            "GPT-4o Mean": round(gap["gpt4o_gap_mean"], 4),
            "GPT-4o-mini Mean": round(gap["mini_gap_mean"], 4),
            "Spearman rho": round(gap.get("gap_spearman_rho", float("nan")), 4),
            "MAD": "",
            "Cohen's d": "",
            "p (BH)": "",
        })

    # Reversed control section
    for subset_name in ["divergent", "control"]:
        if subset_name not in reversed_control:
            continue
        rc = reversed_control[subset_name]
        label = subset_name.title()

        table_rows.append({
            "Experiment": "Reversed Control",
            "Comparison": f"{label}: Literal vs Pref",
            "GPT-4o Mean": f"{rc['literal_mean']:.4f} vs {rc['pref_mean']:.4f}",
            "GPT-4o-mini Mean": "",
            "Spearman rho": "",
            "MAD": "",
            "Cohen's d": round(rc["literal_vs_pref"]["cohens_d"], 4),
            "p (BH)": round(rc["literal_vs_pref"].get("wilcoxon_p_bh", rc["literal_vs_pref"]["wilcoxon_p_raw"]), 4),
        })
        table_rows.append({
            "Experiment": "Reversed Control",
            "Comparison": f"{label}: Literal vs Need",
            "GPT-4o Mean": f"{rc['literal_mean']:.4f} vs {rc['need_mean']:.4f}",
            "GPT-4o-mini Mean": "",
            "Spearman rho": "",
            "MAD": "",
            "Cohen's d": round(rc["literal_vs_need"]["cohens_d"], 4),
            "p (BH)": round(rc["literal_vs_need"].get("wilcoxon_p_bh", rc["literal_vs_need"]["wilcoxon_p_raw"]), 4),
        })

    table_df = pd.DataFrame(table_rows)

    # Statistical notes
    results["statistical_notes"] = {
        "judge_agreement": "Spearman/Pearson correlations between GPT-4o and GPT-4o-mini S_need scores",
        "cohens_kappa": "Binary agreement at median and 0.8 thresholds",
        "reversed_control_p": "Wilcoxon signed-rank paired tests, BH-FDR corrected",
        "reversed_control_d": "Cohen's d_z for paired effect size",
        "cis": "95% clustered bootstrap (10,000 resamples, clustered by person_id)",
        "scoring_matrix": {
            "response_types": ["preference_matched", "need_aware", "literal"],
            "judges": ["gpt-4o", "gpt-4o-mini"],
        },
    }

    return results, table_df


# ─── Main orchestrator ───────────────────────────────────────────────────────

async def main():
    load_dotenv()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        exit(1)

    config = load_config()
    cfg = config["step_7"]
    output_cfg = cfg["output"]
    judge_template = load_judge_prompt(config)

    base_dir = Path(__file__).parent
    data_dir = base_dir / "data"
    results_dir = base_dir / "results"
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    scored_path = base_dir / output_cfg["scored_parquet"]

    print("=" * 80)
    print("Step 7: Judge Robustness — Multi-Judge Agreement & Reversed Control")
    print("=" * 80)

    # ── Load data ──
    if scored_path.exists():
        print(f"\nLoading checkpoint from {scored_path}")
        df = pd.read_parquet(scored_path)
        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")
    else:
        input_path = base_dir / cfg["input_file"]
        if not input_path.exists():
            print(f"ERROR: Input file not found at {input_path}")
            print("Please run step 4 first!")
            exit(1)
        df = pd.read_parquet(input_path)
        print(f"\nLoaded {len(df)} rows from {input_path}")

    client = AsyncOpenAI(api_key=api_key)

    # ── Stage 1: Generate literal responses ──
    print(f"\n{'='*80}")
    print("[Stage 1] Generate literal responses with GPT-4o")
    print("=" * 80)

    df = await stage_1_generate_literal(client, df, config)
    df.to_parquet(scored_path, index=False)
    print(f"  Checkpoint saved to {scored_path}")

    # ── Stage 2: Score with GPT-4o-mini ──
    print(f"\n{'='*80}")
    print("[Stage 2] Score all response types with GPT-4o-mini")
    print("=" * 80)

    df = await stage_2_score_with_mini(client, df, config, judge_template)
    df.to_parquet(scored_path, index=False)
    print(f"  Checkpoint saved to {scored_path}")

    # ── Stage 3: Score literal with GPT-4o ──
    print(f"\n{'='*80}")
    print("[Stage 3] Score literal responses with GPT-4o")
    print("=" * 80)

    df = await stage_3_score_literal_gpt4o(client, df, config, judge_template)
    df.to_parquet(scored_path, index=False)
    print(f"  Checkpoint saved to {scored_path}")

    # ── Stage 4: Compute metrics & output ──
    print(f"\n{'='*80}")
    print("[Stage 4] Computing agreement metrics & output")
    print("=" * 80)

    results, table_df = stage_4_compute_metrics(df)

    if not results:
        print("  Stage 4 failed — no complete rows found")
        return

    # Save results JSON
    json_path = base_dir / output_cfg["results_json"]
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {json_path}")

    # Save Table 5 CSV
    csv_path = base_dir / output_cfg["results_csv"]
    table_df.to_csv(csv_path, index=False)
    print(f"  Table 5 saved to: {csv_path}")

    # ── Print summary ──
    print(f"\n{'='*80}")
    print("Table 5: Judge Robustness Summary")
    print(f"{'='*80}")
    print(table_df.to_string(index=False))

    print(f"\n{'='*80}")
    print("Summary Verdicts")
    print(f"{'='*80}")

    summary = results.get("summary", {})

    rho = summary.get("multi_judge_spearman_rho", float("nan"))
    strong = summary.get("multi_judge_agreement_strong", False)
    print(f"\n  Multi-judge Spearman rho: {rho:.4f} "
          f"({'STRONG' if strong else 'moderate/weak'} agreement)")

    if "divergent_gap_gpt4o" in summary:
        print(f"\n  Divergent S_need gap:")
        print(f"    GPT-4o judge:      {summary['divergent_gap_gpt4o']:+.4f}")
        print(f"    GPT-4o-mini judge: {summary['divergent_gap_mini']:+.4f}")
        reproduces = summary.get("mini_reproduces_divergent_gap", False)
        print(f"    Mini reproduces gap: {'YES' if reproduces else 'NO'}")

    if "control_gap_gpt4o" in summary:
        print(f"\n  Control S_need gap:")
        print(f"    GPT-4o judge:      {summary['control_gap_gpt4o']:+.4f}")
        print(f"    GPT-4o-mini judge: {summary['control_gap_mini']:+.4f}")

    if "content_sensitive_not_origin_sensitive" in summary:
        content_sens = summary["content_sensitive_not_origin_sensitive"]
        print(f"\n  Reversed control (divergent):")
        print(f"    Literal vs Pref d: {summary['literal_vs_pref_d_divergent']:+.4f} "
              f"(similar: {summary['literal_similar_to_pref']})")
        print(f"    Literal vs Need d: {summary['literal_vs_need_d_divergent']:+.4f} "
              f"(different: {summary['literal_different_from_need']})")
        print(f"    Content-sensitive, not origin-sensitive: "
              f"{'YES' if content_sens else 'NO'}")

    print(f"\n{'='*80}")
    print("Judge robustness analysis complete!")
    print(f"{'='*80}")


if __name__ == "__main__":
    asyncio.run(main())
