import asyncio
import json
import pandas as pd
import yaml
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_taxonomy_prompt():
    """Load the taxonomy classifier prompt template"""
    prompt_path = Path(__file__).parent / "prompts" / "taxonomy_classifier.txt"
    with open(prompt_path, 'r') as f:
        return f.read()

async def classify_prompt_async(client, prompt_text, taxonomy_template, model="gpt-4o", semaphore=None):
    """Classify a single prompt using GPT-4o asynchronously"""
    filled_prompt = taxonomy_template.replace("{prompt}", prompt_text)

    try:
        if semaphore:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are an expert at analyzing user queries for query-need divergence."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"}
                )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing user queries for query-need divergence."},
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=0.3,
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

async def filter_prompts_async(config, num_prompts=500, confidence_threshold=0.7, max_concurrent=20):
    """Filter prompts from PersonalLLM dataset asynchronously"""
    # Initialize OpenAI async client
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load data
    data_dir = Path(__file__).parent.parent.parent / "PersonalLLM" / "PersonalLLM" / "data"

    print("Loading PersonalLLM_Eval dataset.")
    combined_df = pd.read_parquet(data_dir / "personal_llm_eval.parquet")
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total prompts available: {len(combined_df)}")

    # Load taxonomy template
    taxonomy_template = load_taxonomy_prompt()

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
    batch_size = 100
    for batch_start in range(0, len(valid_prompts), batch_size):

        batch_end = min(batch_start + batch_size, len(valid_prompts))
        batch = valid_prompts[batch_start:batch_end]

        print(f"\nProcessing batch {batch_start//batch_size + 1} ({batch_start}-{batch_end})...")

        # Create tasks
        tasks = [
            classify_prompt_async(client, prompt_text, taxonomy_template, semaphore=semaphore)
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
                if len(divergent_prompts) < 3:
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
                if len(non_divergent_prompts) < 3:
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

    # Combine divergent and non-divergent prompts
    all_prompts = divergent_prompts + non_divergent_prompts
    filtered_df = pd.DataFrame(all_prompts)

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
    output_path = Path(__file__).parent / "data" / "filtered_prompts.parquet"
    output_path.parent.mkdir(exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

async def main():
    # Load environment variables from .env file
    load_dotenv()

    config = load_config()

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        exit(1)

    _ = await filter_prompts_async(
        config,
        num_prompts=config['num_prompts'],
        confidence_threshold=config['divergence_confidence_threshold'],
        max_concurrent=20  # Adjust based on your rate limits
    )

if __name__ == "__main__":
    asyncio.run(main())
