# README

This pipeline measures whether preference-optimized AI responses align with users' actual needs. It compares PersonalLLM's preference-matched responses against GPT-4o's need-aware responses using reward models and a need-alignment judge.

**Key Finding:** Preference-matched responses score 3.6 points higher on preference rewards but may score lower on need-alignment.

---

## Data Flow

```
PersonalLLM Test Split (1,000 prompts)
  ↓
[Step 1] GPT-4o classifies 985 prompts into divergence types
  ↓
[Step 2] Generate 2 responses per prompt:
         • PersonalLLM's preference-matched response
         • GPT-4o's need-aware response
  ↓
[Step 3] Score 1,970 responses with 8 reward models
  ↓
[Step 4] Statistical analysis & tables
  ↓
[Step 5] GPT-4o judges need-alignment
  ↓
[Step 4 again] Final analysis with complete metrics
```

---

## Quick Execution

```bash
export OPENAI_API_KEY='your-key'

uv run python 1_filter_prompts_async.py
uv run python 2_generate_responses_async.py
uv run python 3_score_responses.py # Requires GPU
uv run python 4_analyze_results.py
uv run python 5_score_need_alignment_async.py
uv run python 4_analyze_results.py
```
---

## Data

**Source:** PersonalLLM test split, shuffled with seed=42

**Selection criteria (all must be met):**
- Confidence ≥ 0.7 from GPT-4o classifier
- Matches target domains
- Either divergent OR non-divergent control

**Final distribution (985 total):**
- **501 divergent (50.9%):**
  - 412 underspecified_intent (41.8%)
  - 78 pedagogical_redirection (7.9%)
  - 9 sycophantic_affirmation (0.9%)
  - 2 dynamic_state_mismatch (0.2%)
- **484 non-divergent controls (49.1%)** - labeled "none"

---

## Divergence Taxonomy

1. **Pedagogical Redirection:** User asks for answer, needs explanation
2. **Sycophantic Affirmation:** User seeks validation, needs honest feedback
3. **Dynamic State Mismatch:** User's state makes preference suboptimal
4. **Underspecified Intent:** Query is ambiguous
5. **None:** No divergence (control group)

---

## Prompts Used

### Step 1: Taxonomy Classifier
- Model: GPT-4o, temp=0.3
- Input: User query
- Output: JSON with divergence_type, confidence, reasoning, domains

### Step 2: Need-Aware Generator
- Model: GPT-4o, temp=0.7
- Input: User query + 3 previous interactions
- Output: Analysis + need-aware response

### Step 5: Need-Alignment Judge
- Model: GPT-4o, temp=0.3
- Input: Query + divergence type + response
- Output: 3 scores (0-5 each) + reasoning
  - Need fulfillment
  - Expert approval
  - Appropriate redirection
- **S_need** = sum of 3 scores / 15 (normalized to 0-1)

---

## Metrics

### R_pref (Preference Reward)
- From 8 reward models
- Mean across all models
- **Higher = users prefer this response**

### S_need (Need-Alignment Score)
- From GPT-4o judge
- Average of 3 criteria (0-5 each), normalized
- **Higher = better addresses underlying need**

### Gaps
- **reward_gap** = pref_reward_mean - need_aware_reward_mean
- **s_need_gap** = need_s_need - pref_s_need

---

## Results Snapshot

**Preference Rewards (R_pref):**
- Preference-matched: 4.076
- Need-aware: 0.447
- **Gap: +3.629** (preference wins 99.7% of time)

**Need-Alignment (S_need) - CRITICAL: Split by Divergence:**

**Divergent Cases (n=501):**
- Preference-matched: 0.882
- Need-aware: 0.964
- **Gap: +0.082** (need-aware wins 48.3%, ties 41.7%, pref wins 10.0%)

**Control Cases (n=484):**
- Preference-matched: 0.994
- Need-aware: 0.991
- **Gap: -0.004** (need-aware wins 0.8%, ties 96.3%, pref wins 2.9%)

**Interpretation:** Reward models strongly prefer preference-matched responses (trained on preferences). But for truly divergent cases, need-aware responses better fulfill underlying needs (+0.082 S_need). Controls show both approaches work equally well, validating the taxonomy.

---

## Key Files

### Config
- `config.yaml` - All parameters

### Prompts (templates)
- `prompts/taxonomy_classifier.txt`
- `prompts/need_aware_generator.txt`
- `prompts/need_alignment_judge.txt`

### Data
- `data/filtered_prompts.parquet` - After Step 1
- `data/generated_responses.parquet` - After Step 2
- `data/scored_responses.parquet` - After Step 3
- `data/fully_scored_responses.parquet` - After Step 5

### Results
- `results/table_2.csv` - Preference vs need comparison
- `results/table_3.csv` - Gaps by divergence type
- `results/qualitative_examples.json` - Sample responses

---

## Research Implications

**Central hypothesis:** Current preference optimization may create a preference-need gap.

**Evidence found:**
1. Preference-matched responses score higher on R_pref (confirmed: 4.076 vs 0.447, gap +3.629)
2. Need-aware responses score higher on S_need (confirmed: 0.977 vs 0.937, gap +0.040)
3. Gap varies by divergence type (confirmed: Underspecified +0.094, Sycophantic +0.089, Control -0.004)
