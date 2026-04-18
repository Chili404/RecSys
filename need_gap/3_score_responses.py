import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXConfig, GPTNeoXModel, GPTNeoXPreTrainedModel
)
import torch.nn as nn
from dataclasses import dataclass
from transformers.utils import ModelOutput
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings('ignore')

# --- Custom model registration for OpenAssistant pythia ---

class GPTNeoXRewardModelConfig(GPTNeoXConfig):
    model_type = "gpt_neox_reward_model"

@dataclass
class GPTNeoXRewardModelOutput(ModelOutput):
    logits: torch.FloatTensor = None

class GPTNeoXRewardModel(GPTNeoXPreTrainedModel):
    config_class = GPTNeoXRewardModelConfig

    def __init__(self, config):
        if isinstance(config, GPTNeoXConfig):
            config = GPTNeoXRewardModelConfig.from_dict(config.to_dict())
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.gpt_neox(input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = outputs[0]
        pooled = hidden_states[:, -1]
        logits = self.out_proj(pooled)
        return GPTNeoXRewardModelOutput(logits=logits)

AutoConfig.register("gpt_neox_reward_model", GPTNeoXRewardModelConfig)
AutoModelForSequenceClassification.register(GPTNeoXRewardModelConfig, GPTNeoXRewardModel)

# --- Config loading ---

config_path = Path(__file__).parent / "config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)['step_3']

REWARD_MODELS = config['reward_models']

print("Configuration loaded from config.yaml")
print(f"Available reward models: {list(REWARD_MODELS.keys())}")

# PATHS AND DATA LOADING

INPUT_PATH = "./data/generated_responses.parquet"
OUTPUT_PATH = "./data/scored_responses.parquet"

# Scoring configuration from config.yaml
MAX_LENGTH = config['scoring']['max_length']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet(INPUT_PATH)

print(f"   Loaded {len(df)} prompts x 2 response types = {len(df) * 2} responses to score")
print(f"   Divergence distribution: {df['divergence_type'].value_counts().to_dict()}")
print(f"Device: {DEVICE}")
print(f"Number of models to score: {len(REWARD_MODELS)}")

# SCORING FUNCTIONS

def format_prompt_response(prompt, response, tokenizer):
    """Format prompt and response using PersonalLLM's convention."""
    if tokenizer.chat_template is not None:
        try:
            formatted = tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            if tokenizer.bos_token:
                formatted = formatted.replace(tokenizer.bos_token, "")
            return formatted
        except Exception:
            pass
    return f"User: {prompt}\nAssistant: {response}"


def score_both_response_types(model_name, prompts, pref_responses, need_responses):
    """
    EFFICIENT: Load model once, score both response types, then unload.
    This cuts runtime in half compared to loading each model twice.
    """
    print(f"   Loading {model_name}...")

    try:
        from transformers import pipeline

        model_config = REWARD_MODELS[model_name]
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['tokenizer_name'],
            trust_remote_code=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            model_config['model_name'],
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        )
        model.eval()

        # Set up padding
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        batch_size = 32
        pipe_kwargs = {
            "batch_size": batch_size,
            "truncation": True,
            "padding": "longest",
            "max_length": MAX_LENGTH,
            "function_to_apply": "none",
            "return_token_type_ids": False,
        }

        reward_pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=DEVICE
        )

        def score_batch(responses, desc):
            formatted = [format_prompt_response(p, r, tokenizer) for p, r in zip(prompts, responses)]  # noqa: F821
            scores = []
            for i in tqdm(range(0, len(formatted), batch_size), desc=f"     {desc}"):
                batch_texts = formatted[i:i+batch_size]
                outputs = reward_pipe(batch_texts, **pipe_kwargs)  # noqa: F821

                if isinstance(outputs[0], dict):
                    batch_scores = [o["score"] for o in outputs]
                elif isinstance(outputs[0][0], dict):
                    batch_scores = [o[0]["score"] for o in outputs]
                else:
                    batch_scores = [float(o[0].cpu().numpy()) if hasattr(o[0], 'cpu') else float(o[0]) for o in outputs]
                scores.extend(batch_scores)
            return scores

        # Score both types (model stays in memory - efficient!)
        print("     Scoring preference-matched...")
        pref_scores = score_batch(pref_responses, "Pref")

        print("     Scoring need-aware...")
        need_scores = score_batch(need_responses, "Need")

        # Cleanup
        del model, tokenizer, reward_pipe
        torch.cuda.empty_cache()

        print(f"     Pref: mean={np.mean(pref_scores):.3f}, Need: mean={np.mean(need_scores):.3f}")
        return pref_scores, need_scores

    except Exception as e:
        import traceback
        print(f"     Error: {e}")
        print(traceback.format_exc())
        return [0.0] * len(prompts), [0.0] * len(prompts)



def main():
    """Main execution function."""

    print("="*80)
    print("EFFICIENT SCORING: Each model loaded once for both response types")
    print(f"Models: {len(REWARD_MODELS)} | Responses: {len(df)} x 2 = {len(df)*2}")
    print("="*80 + "\n")

    prompts = df['test_prompt'].tolist()
    pref_responses = df['preference_matched_response'].tolist()
    need_responses = df['need_aware_response'].tolist()

    all_pref_scores = {}
    all_need_scores = {}

    for idx, model_name in enumerate(REWARD_MODELS, 1):
        print(f"Model {idx}/{len(REWARD_MODELS)}: {model_name}")

        pref_scores, need_scores = score_both_response_types(
            model_name, prompts, pref_responses, need_responses
        )

        all_pref_scores[f'preference_reward_{model_name}'] = pref_scores
        all_need_scores[f'need_aware_reward_{model_name}'] = need_scores
        print()

    print("="*80)
    print("All models scored!")
    print("="*80)

    # Compute means with z-score normalization (test-set method)
    print("\nApplying test-set z-score normalization and computing means...")
    print("  (Normalizing each model using statistics from our test set)")

    # Collect model score columns
    pref_cols = list(all_pref_scores.keys())
    need_cols = list(all_need_scores.keys())

    normalized_pref_scores = {}
    normalized_need_scores = {}

    # Align columns by model name instead of relying on dict order
    pref_models = {
        col.replace("preference_reward_", ""): col
        for col in pref_cols
        if col.startswith("preference_reward_")
    }
    need_models = {
        col.replace("need_aware_reward_", ""): col
        for col in need_cols
        if col.startswith("need_aware_reward_")
    }

    common_models = sorted(set(pref_models) & set(need_models))

    if not common_models:
        raise ValueError("No matching preference/need-aware model columns found.")

    missing_in_need = sorted(set(pref_models) - set(need_models))
    missing_in_pref = sorted(set(need_models) - set(pref_models))

    if missing_in_need:
        print(f"Warning: missing need-aware columns for models: {missing_in_need}")
    if missing_in_pref:
        print(f"Warning: missing preference columns for models: {missing_in_pref}")

    # Normalize each matched model using combined TEST SET statistics
    for model_name in common_models:
        pref_col = pref_models[model_name]
        need_col = need_models[model_name]

        pref_arr = np.asarray(all_pref_scores[pref_col], dtype=float)
        need_arr = np.asarray(all_need_scores[need_col], dtype=float)

        if len(pref_arr) != len(prompts):
            raise ValueError(
                f"{pref_col} has length {len(pref_arr)}, expected {len(prompts)}"
            )
        if len(need_arr) != len(prompts):
            raise ValueError(
                f"{need_col} has length {len(need_arr)}, expected {len(prompts)}"
            )

        combined_scores = np.concatenate([pref_arr, need_arr])
        test_mean = combined_scores.mean()
        test_sd = combined_scores.std()

        # Guard against divide-by-zero
        if test_sd == 0:
            normalized_pref_scores[pref_col] = np.zeros_like(pref_arr).tolist()
            normalized_need_scores[need_col] = np.zeros_like(need_arr).tolist()
            print(
                f"    {model_name:25s} - Test mean: {test_mean:6.2f}, "
                f"Test sd: {test_sd:5.2f} (constant scores; normalized to 0)"
            )
        else:
            normalized_pref_scores[pref_col] = (
                (pref_arr - test_mean) / test_sd
            ).tolist()
            normalized_need_scores[need_col] = (
                (need_arr - test_mean) / test_sd
            ).tolist()
            print(
                f"    {model_name:25s} - Test mean: {test_mean:6.2f}, "
                f"Test sd: {test_sd:5.2f}"
            )

    # Use only matched/aligned columns for the mean aggregation
    aligned_pref_cols = [pref_models[m] for m in common_models]
    aligned_need_cols = [need_models[m] for m in common_models]

    all_pref_scores["preference_reward_mean"] = [
        float(np.mean([normalized_pref_scores[col][i] for col in aligned_pref_cols]))
        for i in range(len(prompts))
    ]

    all_need_scores["need_aware_reward_mean"] = [
        float(np.mean([normalized_need_scores[col][i] for col in aligned_need_cols]))
        for i in range(len(prompts))
    ]

    print("  Test-set z-score normalization applied")

    # Replace raw scores with normalized scores for matched model columns
    for col in aligned_pref_cols:
        all_pref_scores[col] = normalized_pref_scores[col]

    for col in aligned_need_cols:
        all_need_scores[col] = normalized_need_scores[col]

    # Create DataFrames
    pref_scores_df = pd.DataFrame(all_pref_scores)
    need_scores_df = pd.DataFrame(all_need_scores)

    # Combine with original data
    result_df = pd.concat([df, pref_scores_df, need_scores_df], axis=1)
    result_df['reward_gap'] = result_df['preference_reward_mean'] - result_df['need_aware_reward_mean']

    # Save
    result_df.to_parquet(OUTPUT_PATH, index=False)
    pref_scores_df.to_parquet("./data/individual/pref_scores.parquet", index=False)
    need_scores_df.to_parquet("./data/individual/need_scores.parquet", index=False)

    # Summary statistics
    print("\n" + "="*80)
    print("SCORING SUMMARY")
    print("="*80)
    print(f"\nTotal responses: {len(result_df) * 2}")
    print(f"Models used: {len(common_models)}")

    print("\nPreference-matched rewards:")
    print(f"  Mean: {result_df['preference_reward_mean'].mean():.4f}")
    print(f"  Std:  {result_df['preference_reward_mean'].std(ddof=0):.4f}")

    print("\nNeed-aware rewards:")
    print(f"  Mean: {result_df['need_aware_reward_mean'].mean():.4f}")
    print(f"  Std:  {result_df['need_aware_reward_mean'].std(ddof=0):.4f}")

    print("\nReward gap (pref - need):")
    print(f"  Mean: {result_df['reward_gap'].mean():.4f}")
    print(f"  Std:  {result_df['reward_gap'].std(ddof=0):.4f}")

    pref_wins = (result_df['preference_reward_mean'] > result_df['need_aware_reward_mean']).sum()
    print(f"\nPreference-matched scores higher: {pref_wins}/{len(result_df)} ({100*pref_wins/len(result_df):.1f}%)")

    print("\nBreakdown by divergence type:")
    for div_type in sorted(result_df['divergence_type'].unique()):
        subset = result_df[result_df['divergence_type'] == div_type]
        print(f"\n  {div_type} (n={len(subset)}):")
        print(f"    Pref: {subset['preference_reward_mean'].mean():.4f}")
        print(f"    Need: {subset['need_aware_reward_mean'].mean():.4f}")
        print(f"    Gap:  {subset['reward_gap'].mean():+.4f}")

    print(f"\nResults saved to: {OUTPUT_PATH}")
    print("="*80)



if __name__ == "__main__":
    print("\n" + "="*80)
    print("EFFICIENT REWARD MODEL SCORING")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Models to score: {len(REWARD_MODELS)}")
    print(f"Prompts: {len(df)}")
    print("="*80 + "\n")

    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
