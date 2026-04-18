import asyncio
import json
import pandas as pd
import yaml
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv
from collections import Counter

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_taxonomy_prompt(config):
    """Load the taxonomy classifier prompt template"""
    prompt_path = Path(__file__).parent / config['taxonomy_prompt']
    with open(prompt_path, 'r') as f:
        return f.read()

async def classify_prompt_async(client, prompt_text, taxonomy_template,
                                model, system_prompt, temperature, semaphore=None):
    """Classify a single prompt using GPT-4o asynchronously"""
    filled_prompt = taxonomy_template.replace("{prompt}", prompt_text)

    try:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )

        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {
            "has_divergence": False,
            "divergence_type": "none",
            "confidence": 0.0,
            "reasoning": f"Error: {str(e)}",
            "target_domains": []
        }

def compute_class_gaps(divergent_prompts, config):
    """Count current samples per divergence type and return gaps up to max_samples_per_type ceiling."""
    max_samples = config['experiment'].get('max_samples_per_type', 250)
    type_counts = Counter(p['divergence_type'] for p in divergent_prompts)
    gaps = {}
    for dtype in config.get('external_datasets', {}).get('wildchat', {}).get('target_types', []):
        current = type_counts.get(dtype, 0)
        if current < max_samples:
            gaps[dtype] = max_samples - current
    return gaps


WILDCHAT_KEYWORDS = {
    'sycophantic_affirmation': [
        "write a positive", "write a convincing",
        "is it wrong", "is this wrong",
        "was I right", "am I right",
        "do you agree", "do you think I should",
        "my boyfriend", "my girlfriend", "my relationship",
        "make it sound", "grade my",
    ],
    'dynamic_state_mismatch': [
        "exhausted", "overwhelmed", "anxious",
        "can't focus", "stressed", "burned out",
        "struggling", "give up", "feel useless",
        "too much", "can't cope", "falling apart",
    ],
    'pedagogical_redirection': [
        "answer to", "solve this", "give me the answer",
        "homework", "exam", "quiz",
    ],
}

JAILBREAK_PATTERNS = [
    "from now on you are going to act as",
    "do anything now",
    "hypothetical story",
    "unfiltered and amoral",
    "ignore previous",
    "this is a fictional setting",
]


def load_wildchat_candidates(config, target_types, max_candidates):
    """Stream WildChat via HuggingFace and return keyword-filtered candidates.

    Applies keyword heuristics to pre-select likely candidates per target type,
    and excludes jailbreak/manipulation prompts.
    Returns list of (user_prompt, assistant_response) tuples.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("WARNING: 'datasets' library not installed. Cannot load WildChat.")
        return []

    wildchat_config = config.get('external_datasets', {}).get('wildchat', {})
    hf_repo = wildchat_config.get('hf_repo', 'allenai/WildChat')
    language_filter = wildchat_config.get('language_filter', 'English')
    max_scan = wildchat_config.get('max_scan_limit', 200_000)
    progress_interval = wildchat_config.get('progress_interval', 50_000)

    print(f"\nLoading WildChat candidates from {hf_repo} (streaming)...")
    print(f"  Target types: {target_types}")
    print(f"  Max candidates: {max_candidates}")

    try:
        dataset = load_dataset(hf_repo, streaming=True, split='train')
    except Exception as e:
        print(f"WARNING: Failed to load WildChat dataset: {e}")
        return []

    candidates = []
    scanned = 0

    for example in dataset:
        scanned += 1
        if scanned > max_scan:
            break
        if len(candidates) >= max_candidates:
            break

        # Filter to English, non-toxic
        if example.get('language', '') != language_filter:
            continue
        if example.get('toxic', False):
            continue

        # Extract first user message and first assistant response
        conversation = example.get('conversation', [])
        if len(conversation) < 2:
            continue

        user_msg = None
        assistant_msg = None
        for turn in conversation:
            role = turn.get('role', '')
            content = turn.get('content', '')
            if role == 'user' and user_msg is None:
                user_msg = content
            elif role == 'assistant' and assistant_msg is None and user_msg is not None:
                assistant_msg = content
                break

        if not user_msg or not assistant_msg:
            continue
        if not user_msg.strip() or not assistant_msg.strip():
            continue

        prompt_lower = user_msg.lower()

        # Exclude jailbreak/manipulation prompts
        if any(jp in prompt_lower for jp in JAILBREAK_PATTERNS):
            continue

        # Apply keyword heuristics — accept if any target type keyword matches
        matched = False
        for dtype in target_types:
            for kw in WILDCHAT_KEYWORDS.get(dtype, []):
                if kw in prompt_lower:
                    matched = True
                    break
            if matched:
                break

        if not matched:
            continue

        candidates.append((user_msg.strip(), assistant_msg.strip()))

        if scanned % progress_interval == 0:
            print(f"  Scanned {scanned} conversations, {len(candidates)} candidates so far...")

    print(f"  Finished: scanned {scanned} conversations, found {len(candidates)} candidates")
    return candidates


async def classify_and_fill_external_gaps(client, config, taxonomy_template, divergent_prompts,
                                          confidence_threshold, max_concurrent):
    """Orchestrate WildChat gap-filling: identify gaps, load candidates, classify, fill.

    Returns list of new prompt dicts with WildChat data and synthetic placeholders.
    """
    gaps = compute_class_gaps(divergent_prompts, config)
    if not gaps:
        print("\nNo divergence class gaps to fill.")
        return []

    print("\nDivergence class gaps detected:")
    for dtype, needed in gaps.items():
        print(f"  {dtype}: {needed} more needed")

    target_types = list(gaps.keys())
    wildchat_config = config.get('external_datasets', {}).get('wildchat', {})
    max_candidates = wildchat_config.get('candidate_sample_size', 5000)

    candidates = load_wildchat_candidates(config, target_types, max_candidates)
    if not candidates:
        print("WARNING: No WildChat candidates found. Gaps remain unfilled.")
        return []

    max_api_calls = wildchat_config.get('max_api_calls', 20000)
    print(f"\nClassifying {len(candidates)} WildChat candidates with GPT-4o...")
    print(f"  API call budget: {max_api_calls}")

    classification_config = config['classification']
    model = classification_config['model']
    system_prompt = classification_config['system_prompt']
    temperature = classification_config['temperature']
    batch_size = classification_config.get('batch_size', 100)
    default_person_weight = wildchat_config.get('default_person_weight', [0.125] * 8)

    semaphore = asyncio.Semaphore(max_concurrent)
    new_prompts = []
    remaining_gaps = dict(gaps)
    wildchat_counter = 0
    total_api_calls = 0

    # Process in batches
    for batch_start in range(0, len(candidates), batch_size):
        # Check if all gaps are filled
        if all(v <= 0 for v in remaining_gaps.values()):
            print("  All gaps filled early!")
            break

        # Enforce API call budget
        if total_api_calls >= max_api_calls:
            print(f"  API call budget exhausted ({total_api_calls}/{max_api_calls}). Stopping.")
            break

        batch_end = min(batch_start + batch_size, len(candidates))
        batch = candidates[batch_start:batch_end]

        tasks = [
            classify_prompt_async(client, user_prompt, taxonomy_template,
                                  model=model, system_prompt=system_prompt,
                                  temperature=temperature, semaphore=semaphore)
            for user_prompt, _ in batch
        ]

        classifications = await async_tqdm.gather(
            *tasks, desc=f"WildChat batch {batch_start // batch_size + 1}"
        )
        total_api_calls += len(batch)

        for (user_prompt, assistant_response), classification in zip(batch, classifications):
            if not isinstance(classification, dict):
                continue

            div_type = classification.get('divergence_type', 'none')
            conf = classification.get('confidence', 0.0)
            has_div = classification.get('has_divergence', False)

            if not has_div or conf < confidence_threshold:
                continue
            if div_type not in remaining_gaps or remaining_gaps[div_type] <= 0:
                continue

            wildchat_counter += 1
            remaining_gaps[div_type] -= 1

            new_prompts.append({
                'prompt_idx': -wildchat_counter,
                'person_id': f'wildchat_{wildchat_counter}',
                'test_prompt': user_prompt,
                'person_weight': default_person_weight,
                'divergence_type': div_type,
                'confidence': conf,
                'reasoning': classification.get('reasoning', ''),
                'target_domains': classification.get('target_domains', []),
                'user_history_length': 0,
                'is_control': False,
                'prompt_1': '',
                'chosen_1': '',
                'prompt_2': '',
                'chosen_2': '',
                'prompt_3': '',
                'chosen_3': '',
                'data_source': 'wildchat',
                'external_preference_response': assistant_response,
            })

            print(f"  [WildChat +{wildchat_counter}] {div_type} (conf={conf:.2f}), "
                  f"remaining: {remaining_gaps[div_type]}")

    print(f"\nWildChat gap-filling complete: added {len(new_prompts)} prompts")
    for dtype in gaps:
        filled = gaps[dtype] - max(remaining_gaps.get(dtype, 0), 0)
        print(f"  {dtype}: filled {filled}/{gaps[dtype]}")
        if remaining_gaps.get(dtype, 0) > 0:
            print(f"    WARNING: {remaining_gaps[dtype]} still needed for {dtype}")

    return new_prompts


async def filter_prompts_async(config, num_prompts=500, confidence_threshold=0.7, max_concurrent=20):
    """Filter prompts from PersonalLLM dataset asynchronously"""
    # Initialize OpenAI async client
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load data from Hugging Face
    from datasets import load_dataset

    hf_repo = config['personalllm_hf_repo']
    random_seed = config['experiment'].get('random_seed', 42)

    print(f"Loading PersonalLLM_Eval dataset from {hf_repo}...")
    combined_df = load_dataset(hf_repo, split='train').to_pandas()
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    print(f"Total prompts available: {len(combined_df)}")

    # Load taxonomy template
    taxonomy_template = load_taxonomy_prompt(config)

    # Classification parameters
    classification_config = config['classification']
    model = classification_config['model']
    system_prompt = classification_config['system_prompt']
    temperature = classification_config['temperature']
    batch_size = classification_config.get('batch_size', 100)
    debug_log_count = config['experiment'].get('debug_log_count', 3)

    # Filter out invalid prompts
    valid_prompts = []
    for idx, row in combined_df.iterrows():
        prompt_text = row['test_prompt']
        if isinstance(prompt_text, str) and prompt_text.strip():
            valid_prompts.append((idx, row, prompt_text))

    print(f"Valid prompts: {len(valid_prompts)}")

    # Create batches for classification
    target_domains = set(config['target_domains'])
    divergent_prompts = []
    non_divergent_prompts = []

    # Collect all divergent and nondivergent cases
    print("\nClassifying prompts asynchronously...")
    print(f"Collecting all divergent and nondivergent cases (confidence >= {confidence_threshold})")
    print(f"Max concurrent requests: {max_concurrent}")
    print(f"Will process all {len(valid_prompts)} valid prompts")

    # Process in batches
    for batch_start in range(0, len(valid_prompts), batch_size):

        batch_end = min(batch_start + batch_size, len(valid_prompts))
        batch = valid_prompts[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end})...")

        # Create tasks
        tasks = [
            classify_prompt_async(client, prompt_text, taxonomy_template,
                                  model=model, system_prompt=system_prompt,
                                  temperature=temperature, semaphore=semaphore)
            for idx, row, prompt_text in batch
        ]

        # Execute batch asynchronously with progress bar
        classifications = await async_tqdm.gather(*tasks, desc=f"Batch {batch_start//batch_size + 1}")

        # Process results
        for (idx, row, prompt_text), classification in zip(batch, classifications):
            # Check if this prompt meets our criteria
            # Skip if classification failed or is missing required fields
            if not isinstance(classification, dict) or 'has_divergence' not in classification:
                continue

            has_target_domain = bool(set(classification.get('target_domains', [])) & target_domains)
            meets_confidence = classification.get('confidence', 0) >= confidence_threshold
            has_divergence = classification.get('has_divergence', False)

            # Collect divergent cases
            if has_divergence and meets_confidence and has_target_domain:
                if len(divergent_prompts) < debug_log_count:
                    print(f"    DEBUG: Adding divergent prompt_idx={idx}, prompt={prompt_text[:50]}...")

                divergent_prompts.append({
                    'prompt_idx': idx,
                    'person_id': row['person_id'],
                    'test_prompt': prompt_text,
                    'person_weight': row['person_weight'],
                    'divergence_type': classification['divergence_type'],
                    'confidence': classification['confidence'],
                    'reasoning': classification['reasoning'],
                    'target_domains': classification['target_domains'],
                    'user_history_length': row['user_history_length'],
                    'is_control': False,
                    'prompt_1': row.get('prompt_1', ''),
                    'chosen_1': row.get('chosen_1', ''),
                    'prompt_2': row.get('prompt_2', ''),
                    'chosen_2': row.get('chosen_2', ''),
                    'prompt_3': row.get('prompt_3', ''),
                    'chosen_3': row.get('chosen_3', ''),
                })

                total = len(divergent_prompts) + len(non_divergent_prompts)
                print(f"[{total}] Divergent: {classification['divergence_type']} (conf={classification['confidence']:.2f})")

            # Collect non-divergent controls
            elif not has_divergence and meets_confidence and has_target_domain:
                if len(non_divergent_prompts) < debug_log_count:
                    print(f"    DEBUG: Adding control prompt_idx={idx}, prompt={prompt_text[:50]}...")

                non_divergent_prompts.append({
                    'prompt_idx': idx,
                    'person_id': row['person_id'],
                    'test_prompt': prompt_text,
                    'person_weight': row['person_weight'],
                    'divergence_type': 'none',
                    'confidence': classification['confidence'],
                    'reasoning': classification['reasoning'],
                    'target_domains': classification['target_domains'],
                    'user_history_length': row['user_history_length'],
                    'is_control': True,
                    'prompt_1': row.get('prompt_1', ''),
                    'chosen_1': row.get('chosen_1', ''),
                    'prompt_2': row.get('prompt_2', ''),
                    'chosen_2': row.get('chosen_2', ''),
                    'prompt_3': row.get('prompt_3', ''),
                    'chosen_3': row.get('chosen_3', ''),
                })

                total = len(divergent_prompts) + len(non_divergent_prompts)
                print(f"[{total}] Control: none (conf={classification['confidence']:.2f})")

    # Tag all PersonalLLM prompts with data_source
    for p in divergent_prompts:
        p['data_source'] = 'personalllm'
        p['external_preference_response'] = None
    for p in non_divergent_prompts:
        p['data_source'] = 'personalllm'
        p['external_preference_response'] = None

    # Fill underrepresented divergence classes with WildChat
    wildchat_prompts = await classify_and_fill_external_gaps(
        client, config, taxonomy_template, divergent_prompts,
        confidence_threshold, max_concurrent
    )
    divergent_prompts.extend(wildchat_prompts)

    # Combine divergent and non-divergent prompts
    all_prompts = divergent_prompts + non_divergent_prompts
    filtered_df = pd.DataFrame(all_prompts)

    # Ensure person_id is consistently string (PersonalLLM uses int, WildChat uses str)
    filtered_df['person_id'] = filtered_df['person_id'].astype(str)

    # Print statistics
    print("\n" + "="*80)
    print("Filtering complete!")
    print("="*80)
    print(f"Total prompts filtered: {len(filtered_df)}")
    print(f"  Divergent cases: {len(divergent_prompts)}")
    print(f"  Non-divergent controls: {len(non_divergent_prompts)}")
    print("\nDivergence type distribution:")
    print(filtered_df['divergence_type'].value_counts())
    print(f"\nAverage confidence: {filtered_df['confidence'].mean():.3f}")
    print("\nDomain distribution:")
    all_domains = [d for domains in filtered_df['target_domains'] for d in domains]
    domain_counts = pd.Series(all_domains).value_counts()
    print(domain_counts)

    # Save to file
    output_path = Path(__file__).parent / config['output_file']
    output_path.parent.mkdir(exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

async def main():
    # Load environment variables from .env file
    load_dotenv()

    config = load_config()['step_1']

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        exit(1)

    _ = await filter_prompts_async(
        config,
        num_prompts=config['experiment']['num_prompts'],
        confidence_threshold=config['experiment']['divergence_confidence_threshold'],
        max_concurrent=config['classification']['max_concurrent'],
    )

if __name__ == "__main__":
    asyncio.run(main())
