import sys
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import numpy as np
from tqdm import tqdm
import warnings
import yaml
warnings.filterwarnings('ignore')

# Load configuration from config.yaml
config_path = Path('config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

REWARD_MODELS = config['reward_models']

print("✓ Configuration loaded from config.yaml")
print(f"Available reward models: {list(REWARD_MODELS.keys())}")

# PATHS AND DATA LOADING

INPUT_PATH = "./data/generated_responses.parquet"
OUTPUT_PATH = "./data/scored_responses.parquet"

# Scoring configuration from config.yaml
MAX_LENGTH = config['scoring']['max_length']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_parquet(INPUT_PATH)

print(f"   Loaded {len(df)} prompts × 2 response types = {len(df) * 2} responses to score")
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
            formatted = [format_prompt_response(p, r, tokenizer) for p, r in zip(prompts, responses)]
            scores = []
            for i in tqdm(range(0, len(formatted), batch_size), desc=f"     {desc}"):
                batch_texts = formatted[i:i+batch_size]
                outputs = reward_pipe(batch_texts, **pipe_kwargs)
                
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
        
        print(f"     ✓ Pref: mean={np.mean(pref_scores):.3f}, Need: mean={np.mean(need_scores):.3f}")
        return pref_scores, need_scores
        
    except Exception as e:
        import traceback
        print(f"     ✗ Error: {e}")
        print(traceback.format_exc())
        return [0.0] * len(prompts), [0.0] * len(prompts)



def main():
    """Main execution function."""
    
    print("="*80)
    print("EFFICIENT SCORING: Each model loaded once for both response types")
    print(f"Models: {len(REWARD_MODELS)} | Responses: {len(df)} × 2 = {len(df)*2}")
    print("="*80 + "\n")
    
    prompts = df['test_prompt'].tolist()
    pref_responses = df['preference_matched_response'].tolist()
    need_responses = df['need_aware_response'].tolist()
    
    all_pref_scores = {}
    all_need_scores = {}
    
    for idx, model_name in enumerate(REWARD_MODELS, 1):
        print(f"📊 Model {idx}/{len(REWARD_MODELS)}: {model_name}")
        
        pref_scores, need_scores = score_both_response_types(
            model_name, prompts, pref_responses, need_responses
        )
        
        all_pref_scores[f'preference_reward_{model_name}'] = pref_scores
        all_need_scores[f'need_aware_reward_{model_name}'] = need_scores
        print()
    
    print("="*80)
    print("✓ All models scored!")
    print("="*80)
    
    
    
    # Compute means
    print("\nComputing mean scores across all models...")
    
    pref_cols = list(all_pref_scores.keys())
    all_pref_scores['preference_reward_mean'] = [
        np.mean([all_pref_scores[col][i] for col in pref_cols])
        for i in range(len(prompts))
    ]
    
    need_cols = list(all_need_scores.keys())
    all_need_scores['need_aware_reward_mean'] = [
        np.mean([all_need_scores[col][i] for col in need_cols])
        for i in range(len(prompts))
    ]
    
    # Create DataFrames
    pref_scores_df = pd.DataFrame(all_pref_scores)
    need_scores_df = pd.DataFrame(all_need_scores)
    
    # Combine with original data
    result_df = pd.concat([df, pref_scores_df, need_scores_df], axis=1)
    result_df['reward_gap'] = result_df['preference_reward_mean'] - result_df['need_aware_reward_mean']
    
    # Save
    result_df.to_parquet(OUTPUT_PATH, index=False)
    pref_scores_df.to_parquet("./data/pref_scores.parquet")
    need_scores_df.to_parquet("./data/need_scores.parquet")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SCORING SUMMARY")
    print("="*80)
    print(f"\nTotal responses: {len(result_df) * 2}")
    print(f"Models used: {len(REWARD_MODELS)}")
    
    print("\nPreference-matched rewards:")
    print(f"  Mean: {result_df['preference_reward_mean'].mean():.4f}")
    print(f"  Std:  {result_df['preference_reward_mean'].std():.4f}")
    
    print("\nNeed-aware rewards:")
    print(f"  Mean: {result_df['need_aware_reward_mean'].mean():.4f}")
    print(f"  Std:  {result_df['need_aware_reward_mean'].std():.4f}")
    
    print("\nReward gap (pref - need):")
    print(f"  Mean: {result_df['reward_gap'].mean():.4f}")
    print(f"  Std:  {result_df['reward_gap'].std():.4f}")
    
    pref_wins = (result_df['preference_reward_mean'] > result_df['need_aware_reward_mean']).sum()
    print(f"\nPreference-matched scores higher: {pref_wins}/{len(result_df)} ({100*pref_wins/len(result_df):.1f}%)")
    
    print("\nBreakdown by divergence type:")
    for div_type in sorted(result_df['divergence_type'].unique()):
        subset = result_df[result_df['divergence_type'] == div_type]
        print(f"\n  {div_type} (n={len(subset)}):")
        print(f"    Pref: {subset['preference_reward_mean'].mean():.4f}")
        print(f"    Need: {subset['need_aware_reward_mean'].mean():.4f}")
        print(f"    Gap:  {subset['reward_gap'].mean():+.4f}")
    
    print(f"\n✓ Results saved to: {OUTPUT_PATH}")
    print("="*80)



if __name__ == "__main__":
    print("\n" + "="*80)
    print("EFFICIENT REWARD MODEL SCORING")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    print(f"Models to score: {len(REWARD_MODELS)}")
    print(f"Prompts: {len(df)}")
    print("\nEstimated runtime: 20-30 minutes on GPU")
    print("="*80 + "\n")
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
