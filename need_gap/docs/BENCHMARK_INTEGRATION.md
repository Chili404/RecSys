# Benchmark Integration Plan: NeedGap-Bench

## Source Data

**File:** `bench/NeedGap-Bench.xlsx` (17 MB)

**Sheets:**

| Sheet | Rows | Contents |
|-------|------|----------|
| AITA-YTA | 2,000 | Core AITA dataset with human annotations (validation, indirectness, framing) |
| AITA-YTA_full_results | 2,002 | 12 LLM responses per prompt with per-model evaluation scores |
| AITA-YTA filtered convo | 999 | Conversation-formatted subset for dialogue evaluation |
| filtered wildchat-1k | 1,000 | WildChat conversations with domain/signal metadata |
| Test-annotated-samples | 999 | Human-annotated stated vs. inferred need divergence |
| Train-split | 1,750 | Train/test split mappings |

---

## Integration Strategies

### 1. Human Ground Truth Calibration for S_need Judge

**Problem:** Step 4 uses GPT-4o as the sole need-alignment judge. No human validation exists for this metric.

**Data source:** `Test-annotated-samples` (999 rows with `stated_query`, `inferred_need`, `divergence_type`, `divergence_severity`, `ideal_response_sketch`)

**Approach:**
- Score the 999 annotated samples with the existing `need_alignment_judge.txt` prompt
- Measure correlation between GPT-4o S_need scores and human `divergence_severity` labels
- If correlation is weak, fine-tune the judge prompt or add few-shot examples from human annotations
- Report inter-annotator agreement (GPT-4o vs. human) as a validity metric

**Implementation:**
- New script: `6_validate_judge.py`
- Load annotated samples, run through Step 4 scoring
- Compute Spearman correlation, Cohen's kappa, confusion matrix
- Output: `results/judge_validation.json`

**Impact:** Directly addresses reviewer concerns about judge reliability.

---

### 2. Validation / Indirectness / Framing as Complementary Metrics

**Problem:** S_need aggregates three criteria into one number and may miss specific failure modes like sycophancy.

**Data source:** `AITA-YTA_full_results` has per-model scores for `validation`, `indirectness`, and `framing`.

**Approach:**
- Add a scoring step that evaluates each response on three orthogonal dimensions:
  - **Validation rate** — does the model agree with the user when it shouldn't?
  - **Indirectness** — does the model hedge rather than give direct feedback?
  - **Framing** — does the model reframe the issue appropriately?
- Map to divergence types:
  - `Sycophantic Affirmation` → high validation + high indirectness = failure
  - `Pedagogical Redirection` → low framing = failure
- Report per-divergence-type in Table 3 alongside R_pref and S_need

**Implementation:**
- New script: `3b_score_bench_metrics.py`
- GPT-4o prompt to classify each response on the three binary dimensions
- Extend `5_analyze_results.py` to include these metrics in output tables

**Impact:** More granular understanding of how models fail, not just that they fail.

---

### 3. Multi-Model Generalization

**Problem:** Step 3 tests 8 reward models. The claim that preference optimization creates need gaps could be challenged as model-specific.

**Data source:** `AITA-YTA_full_results` has generations from 12 production models (Claude, GPT-4o, GPT-5, Gemini, Llama-8B/17B/70B, Mistral-7B/24B, Qwen, DeepSeek).

**Approach:**
- Run the existing 8 reward models on the bench's 12 model responses
- Show that all 12 generative models exhibit the same sycophancy pattern (high R_pref, low validation agreement with humans)
- Produce a new Table 4: "Need Gap Across Model Families"
  - Rows: each of the 12 models
  - Columns: R_pref, S_need, Validation agreement with Human, Framing score
- Proves the need gap is a systemic property of preference optimization

**Implementation:**
- New script: `6_cross_model_analysis.py`
- Load all 12 model responses from bench
- Score with reward models (Step 3 logic) and need-alignment judge (Step 4 logic)
- Generate Table 4 with per-model comparison
- Output: `results/table_4_cross_model.csv`

**Impact:** Strongest generalizability argument for the paper.

---

### 4. WildChat Signal Labels as Divergence Validators

**Problem:** Step 1's divergence classification relies entirely on GPT-4o. No external signal validates that divergence actually exists in practice.

**Data source:** `filtered wildchat-1k` has `signal_label` annotations (`reformulation`, `frustration`) with confidence scores — behavioral signals that user needs weren't met.

**Approach:**
- Use signal labels as proxy ground truth for divergence:
  - `reformulation` → user re-asked → likely `Underspecified Intent` or failed need detection
  - `frustration` → user frustrated → likely `Dynamic State Mismatch` or `Sycophantic Affirmation`
- Run Step 1's taxonomy classifier on these 1,000 conversations
- Measure correlation between predicted divergence types and observed behavioral signals
- Report agreement rate as evidence that the taxonomy captures real user dissatisfaction

**Implementation:**
- Extend `6_validate_judge.py` or create `6b_validate_taxonomy.py`
- Load WildChat signal data, classify with taxonomy prompt
- Cross-tabulate: divergence_type × signal_label
- Chi-squared test for independence
- Output: `results/taxonomy_signal_validation.json`

**Impact:** External behavioral validation that divergence taxonomy maps to real-world user friction.

---

### 5. Formal Classifier Evaluation with Train/Test Split

**Problem:** The taxonomy classifier (Step 1) is never formally evaluated for classification accuracy.

**Data source:** `Train-split` (1,750 rows) + `Test-annotated-samples` (999 rows with human `divergence_type` and `severity` labels).

**Approach:**
- Use the 999 test samples as a held-out evaluation for the taxonomy classifier
- Run `taxonomy_classifier.txt` prompt on the 999 test samples
- Compute:
  - Per-type and overall classification accuracy
  - Confusion matrix (predicted vs. human-labeled divergence type)
  - Confidence calibration curve (does 0.7 threshold ≈ 70% accuracy?)
  - Per-type precision, recall, F1
- If accuracy is low for specific types, use training samples as few-shot examples in the classifier prompt

**Implementation:**
- New script: `6_validate_classifier.py`
- Load test-annotated-samples, run taxonomy classifier
- Compare predicted `divergence_type` with human annotations
- Output: `results/classifier_evaluation.json`, `results/confusion_matrix.csv`

**Impact:** Formal classification metrics; demonstrates methodological rigor.

---

## Priority Order

| Priority | Strategy | Effort | Impact | Justification |
|----------|----------|--------|--------|---------------|
| 1 | Human ground truth calibration (#1) | Low | High | Validates core metric |
| 2 | Multi-model generalization (#3) | Medium | High | Proves systemic claim |
| 3 | Classifier evaluation (#5) | Low | Medium | Methodological rigor |
| 4 | WildChat signal validation (#4) | Low | Medium | External validity |
| 5 | Validation/Indirectness/Framing (#2) | High | Medium | Finer granularity |

---

## Config Extensions

Add to `config.yaml`:

```yaml
bench:
  path: "../bench/NeedGap-Bench.xlsx"
  sheets:
    aita_convo: "AITA-YTA filtered convo"
    wildchat: "filtered wildchat-1k"
    test_annotated: "Test-annotated-samples"
    train_split: "Train-split"
  model_columns:
    - Claude
    - DeepSeek
    - GPT-4o
    - GPT-5
    - Gemini
    - Llama-8B
    - Llama-17B
    - Llama-70B
    - Mistral-7B
    - Mistral-24B
    - Qwen
  wildchat:
    signal_labels: ["reformulation", "frustration"]
    min_signal_score: 0.8
  validation:
    test_set_sheet: "Test-annotated-samples"
    confidence_threshold: 0.7
```

---

## New Scripts Summary

| Script | Depends On | Produces |
|--------|-----------|----------|
| `6_validate_classifier.py` | bench xlsx, taxonomy_classifier.txt | `results/classifier_evaluation.json` |
| `6b_validate_taxonomy.py` | bench xlsx (wildchat sheet) | `results/taxonomy_signal_validation.json` |
| `6_cross_model_analysis.py` | bench xlsx (AITA full), reward models | `results/table_4_cross_model.csv` |
| `3b_score_bench_metrics.py` | generated_responses.parquet | Extended scoring with validation/indirectness/framing |

---

## Data Quality Notes

- **Train-split sheet** has misaligned data in some rows (math problems appearing in `split_id` column). Requires cleaning before use.
- **AITA-YTA_full_results** contains 4 duplicate Mistral-7B columns — deduplicate or average.
- **Test-annotated-samples** column `a` is partially filled and likely a draft response column — ignore or verify purpose before use.
