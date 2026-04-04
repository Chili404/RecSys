import asyncio
import pandas as pd
import yaml
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from openai import AsyncOpenAI
import os
import re
from dotenv import load_dotenv

def load_config():
    """Load configuration from config.yaml"""
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_need_aware_prompt():
    """Load the need-aware generator prompt template"""
    prompt_path = Path(__file__).parent / "prompts" / "need_aware_generator.txt"
    with open(prompt_path, 'r') as f:
        return f.read()

def get_user_context(row):
    """Extract user preference history as context"""
    context_parts = []

    for i in range(1, 4):  # Use first 3 interactions for context
        prompt_col = f'prompt_{i}'
        chosen_col = f'chosen_{i}'

        if prompt_col in row and pd.notna(row[prompt_col]) and row[prompt_col]:
            prompt_text = row[prompt_col]
            chosen = row[chosen_col]
            context_parts.append(f"Previous query: {prompt_text}\nUser preferred response: {chosen}")

    if context_parts:
        return "\n\n".join(context_parts)
    return "No previous context available."

async def generate_need_aware_response_async(client, prompt_text, context, template, model="gpt-4o", semaphore=None):
    """Generate a need-aware response using GPT-4o asynchronously."""
    filled_prompt = template.replace("{prompt}", prompt_text).replace("{context}", context)

    try:
        if semaphore:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a need-aware assistant that responds to underlying user needs."},
                        {"role": "user", "content": filled_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
        else:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a need-aware assistant that responds to underlying user needs."},
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )

        content = response.choices[0].message.content

        # Extract analysis and response sections
        analysis_match = re.search(r'<analysis>(.*?)</analysis>', content, re.DOTALL)
        response_match = re.search(r'<response>(.*?)</response>', content, re.DOTALL)

        analysis = analysis_match.group(1).strip() if analysis_match else ""
        response_text = response_match.group(1).strip() if response_match else content

        return {
            "analysis": analysis,
            "response": response_text,
            "full_output": content
        }
    except Exception as e:
        return {
            "analysis": f"Error: {str(e)}",
            "response": "",
            "full_output": ""
        }

async def generate_responses_async(config, max_concurrent=20):
    """Generate preference-matched and need-aware responses asynchronously"""
    # Initialize OpenAI async client
    api_key = os.environ.get("OPENAI_API_KEY")
    client = AsyncOpenAI(api_key=api_key)

    # Semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    # Load filtered prompts
    filtered_path = Path(__file__).parent / "data" / "filtered_prompts.parquet"
    if not filtered_path.exists():
        print(f"ERROR: Filtered prompts not found at {filtered_path}")
        print("Please run 1_filter_prompts.py first!")
        return None

    filtered_df = pd.read_parquet(filtered_path)
    print(f"Loaded {len(filtered_df)} filtered prompts")

    # Load PersonalLLM datasets to get best_response
    data_dir = Path(__file__).parent.parent.parent / "PersonalLLM" / "PersonalLLM" / "data"
    print("Loading PersonalLLM_Eval dataset (matching filter script)...")
    # Use eval dataset (derived from test split) to match 1_filter_prompts_async.py
    combined_df = pd.read_parquet(data_dir / "personal_llm_eval.parquet")
    # IMPORTANT: Must shuffle the same way as 1_filter_prompts_async.py to align indices
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Load need-aware template
    need_aware_template = load_need_aware_prompt()

    print("\nGenerating need-aware responses asynchronously...")
    print(f"Max concurrent requests: {max_concurrent}")

    # Prepare data for async generation
    tasks = []
    rows_data = []

    for idx, row in filtered_df.iterrows():
        prompt_text = row['test_prompt']
        context = get_user_context(row)

        # Create async task
        task = generate_need_aware_response_async(
            client, prompt_text, context, need_aware_template, semaphore=semaphore
        )
        tasks.append(task)
        rows_data.append((idx, row, prompt_text))

    # Execute all tasks asynchronously with progress bar
    results = await async_tqdm.gather(*tasks, desc="Generating responses")

    # Process results and combine with preference-matched responses
    all_results = []

    for (idx, row, prompt_text), need_aware_result in zip(rows_data, results):
        prompt_idx = row['prompt_idx']

        # Get preference-matched response from PersonalLLM data
        # Use .loc[] to access by index value, not .iloc[] for position
        original_row = combined_df.loc[prompt_idx]
        preference_matched_response = original_row['best_response']
        preference_matched_model = original_row.get('best_response_model', 'unknown')
        preference_matched_reward = original_row.get('best_response_reward', 0.0)

        # Store results
        all_results.append({
            'prompt_idx': prompt_idx,
            'person_id': row['person_id'],
            'test_prompt': prompt_text,
            'person_weight': row['person_weight'],
            'divergence_type': row['divergence_type'],
            'confidence': row['confidence'],
            'reasoning': row['reasoning'],
            'target_domains': row['target_domains'],

            # Preference-matched response
            'preference_matched_response': preference_matched_response,
            'preference_matched_model': preference_matched_model,
            'preference_matched_reward': preference_matched_reward,

            # Need-aware response
            'need_aware_response': need_aware_result['response'],
            'need_aware_analysis': need_aware_result['analysis'],

            # User context
            'user_context': get_user_context(row),
        })

    # Create DataFrame
    results_df = pd.DataFrame(all_results)

    # Print statistics
    print("\n" + "="*80)
    print("Response generation complete!")
    print("="*80)
    print(f"Total responses generated: {len(results_df)}")
    print("\nDivergence type distribution:")
    print(results_df['divergence_type'].value_counts())

    # Save to file
    output_path = Path(__file__).parent / "data" / "generated_responses.parquet"
    results_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Also save as JSON for inspection
    json_path = output_path.with_suffix('.json')
    results_df.to_json(json_path, orient='records', indent=2)
    print(f"Also saved as JSON: {json_path}")

    # Save a few examples for qualitative inspection
    sample_path = Path(__file__).parent / "results" / "response_examples.txt"
    sample_path.parent.mkdir(exist_ok=True)

    with open(sample_path, 'w') as f:
        f.write("Sample Generated Responses\n")
        f.write("="*80 + "\n\n")

        for div_type in results_df['divergence_type'].unique():
            samples = results_df[results_df['divergence_type'] == div_type].head(2)

            for idx, row in samples.iterrows():
                f.write(f"Divergence Type: {row['divergence_type'].upper()}\n")
                f.write(f"Confidence: {row['confidence']:.2f}\n")
                f.write(f"Reasoning: {row['reasoning']}\n")
                f.write(f"\nPrompt: {row['test_prompt']}\n")
                f.write("\n--- Preference-Matched Response ---\n")
                f.write(f"{row['preference_matched_response']}\n")
                f.write("\n--- Need-Aware Response ---\n")
                f.write(f"{row['need_aware_response']}\n")
                f.write("\n--- Need-Aware Analysis ---\n")
                f.write(f"{row['need_aware_analysis']}\n")
                f.write("\n" + "="*80 + "\n\n")

    print(f"Sample responses saved to: {sample_path}")

    return results_df

async def main():
    # Load environment variables from .env file
    load_dotenv()

    config = load_config()

    # Check for OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        exit(1)

    _ = await generate_responses_async(config, max_concurrent=20)

if __name__ == "__main__":
    asyncio.run(main())
