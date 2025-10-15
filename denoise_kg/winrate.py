import os
import json
import json_repair
import asyncio
import logging 
from typing import Dict, Any
from openai import AsyncOpenAI, RateLimitError, APIConnectionError
from collections import Counter
import yaml
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from tqdm.asyncio import tqdm_asyncio
import argparse

# --- Evaluation Settings ---
global EVALUATION_MODEL
CONCURRENCY_LIMIT = 32

# --- Prompts (remain the same) ---
SYS_PROMPT = """
---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.
"""
USER_PROMPT_TEMPLATE = """
You will evaluate two answers to the same question based on three criteria: **Comprehensiveness**, **Diversity**, and **Empowerment**.

- **Comprehensiveness**: How much detail does the answer provide to cover all aspects and details of the question?
- **Diversity**: How varied and rich is the answer in providing different perspectives and insights on the question?
- **Empowerment**: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why. Then, select an overall winner based on these three categories.

Here is the question:
{query}

Here are the two answers:

**Answer 1:**
{answer1}

**Answer 2:**
{answer2}

Evaluate both answers using the three criteria listed above and provide detailed explanations for each criterion.

Output your evaluation in the following JSON format:

{{
    "Comprehensiveness": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Diversity": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Empowerment": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Provide explanation here]"
    }},
    "Overall Winner": {{
        "Winner": "[Answer 1 or Answer 2]",
        "Explanation": "[Summarize why this answer is the overall winner based on the three criteria]"
    }}
}}
"""

def load_results_file(filepath: str) -> Dict[int, Dict]:
    data = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    if 'id' in record:
                        data[record['id']] = record
                except json.JSONDecodeError:
                    tqdm_asyncio.write(f"Warning: Could not decode line in {filepath}: {line.strip()}")
    except FileNotFoundError:
        tqdm_asyncio.write(f"ERROR: File not found at {filepath}")
        return None
    return data

# --- Custom logging function for tenacity ---
def log_retry_attempt(retry_state):
    """Custom logging function that only prints on retries and uses tqdm.write."""
    tqdm_asyncio.write(
        f"Retrying API call... Attempt #{retry_state.attempt_number} "
        f"due to: {type(retry_state.outcome.exception()).__name__}"
    )

@retry(
    wait=wait_random_exponential(min=2, max=60),
    stop=stop_after_attempt(20),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
    before_sleep=log_retry_attempt # Use the custom logging function
)
async def evaluate_pair(
    session: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    question_id: int,
    query: str,
    answer1: str,
    answer2: str
) -> Dict[str, Any]:
    async with semaphore:
        try:
            prompt = USER_PROMPT_TEMPLATE.format(query=query, answer1=answer1, answer2=answer2)
            response = await session.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            llm_output = response.choices[0].message.content
            evaluation_json = json_repair.loads(llm_output)
            return {"id": question_id, "question": query, "evaluation": evaluation_json}
        except Exception as e:
            tqdm_asyncio.write(f"--- 🔴 Request Failed (ID: {question_id}) after all retries ---")
            tqdm_asyncio.write(f"    Error Type: {type(e).__name__}")
            # tqdm_asyncio.write(f"    llm_output: {llm_output}")
            tqdm_asyncio.write(f"    Error Details: {e}")
            tqdm_asyncio.write(f"----------------------------------------------------")
            return {"id": question_id, "question": query, "evaluation": {"error": str(e)}}

async def run_evaluation_round(client: AsyncOpenAI, answer1_path: str, answer2_path: str, output_path: str):
    print(f"\n--- Starting Evaluation Round ---")
    print(f"   Answer 1: {answer1_path.replace(os.sep, '/')}")
    print(f"   Answer 2: {answer2_path.replace(os.sep, '/')}")


    data1 = load_results_file(answer1_path)
    data2 = load_results_file(answer2_path)
    if data1 is None or data2 is None: return None, None, None

    common_ids = sorted(list(set(data1.keys()) & set(data2.keys())))
    if not common_ids:
        print(" -> No common questions found for this round.")
        return None, None, None
        
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    tasks = [evaluate_pair(client, semaphore, q_id, data1[q_id]['question'], data1[q_id]['output'], data2[q_id]['output']) for q_id in common_ids]
    results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating Round ({os.path.basename(output_path)})")
    
    criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
    wins = {criterion: Counter() for criterion in criteria}
    valid_evals = {criterion: 0 for criterion in criteria}

    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            eval_data = result.get("evaluation", {})
            for criterion in criteria:
                if criterion in eval_data and "Winner" in eval_data[criterion]:
                    winner = eval_data[criterion]["Winner"]
                    wins[criterion][winner] += 1
                    valid_evals[criterion] += 1
    
    return wins, valid_evals, common_ids

async def main():
    print("--- Starting Debiased LLM-based Win Rate Evaluation ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n[Step 1] Loading configuration from {CONFIG_PATH}...")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    api_key = config.get('llm', {}).get('api_key')
    base_url = config.get('llm', {}).get('llm_base_url')
    model = config.get('llm', {}).get('model')
    global EVALUATION_MODEL
    EVALUATION_MODEL = model

    try:
        client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    except Exception as e:
        print(f"ERROR: Failed to initialize OpenAI client. Details: {e}")
        return

    round1_output = os.path.join(OUTPUT_DIR, "round1_eval_results.jsonl")
    wins1, valid_evals1, common_ids_round1 = await run_evaluation_round(client, FILE_1_PATH, FILE_2_PATH, round1_output)
    
    round2_output = os.path.join(OUTPUT_DIR, "round2_eval_results.jsonl")
    wins2, valid_evals2, _ = await run_evaluation_round(client, FILE_2_PATH, FILE_1_PATH, round2_output)

    if wins1 is None or wins2 is None:
        print("\nEvaluation could not be completed.")
        return

    print("\n\n--- Final Debiased Evaluation Report ---")
    
    total_requests = len(common_ids_round1) * 2
    total_valid_overall = valid_evals1.get("Overall Winner", 0) + valid_evals2.get("Overall Winner", 0)
    failed_requests = total_requests - total_valid_overall
    
    print(f"Total question pairs processed: {len(common_ids_round1)}")
    if failed_requests > 0:
        print(f"🔴 Total failed requests (after all retries): {failed_requests}")
    
    criteria = ["Comprehensiveness", "Diversity", "Empowerment", "Overall Winner"]
    
    def print_final_win_rate(criterion_name):
        print(f"\n----- {criterion_name} -----")
        
        file1_wins = wins1[criterion_name].get("Answer 1", 0) + wins2[criterion_name].get("Answer 2", 0)
        file2_wins = wins1[criterion_name].get("Answer 2", 0) + wins2[criterion_name].get("Answer 1", 0)
        total_ties = wins1[criterion_name].get("Tie", 0) + wins2[criterion_name].get("Tie", 0)
        total_valid = valid_evals1.get(criterion_name, 0) + valid_evals2.get(criterion_name, 0)
        
        if total_valid == 0:
            print("No valid evaluations for this criterion.")
            return

        win_rate1 = (file1_wins / total_valid) * 100
        win_rate2 = (file2_wins / total_valid) * 100
        tie_rate = (total_ties / total_valid) * 100
        
        print(f"File 1 Wins: {file1_wins} ({win_rate1:.2f}%)")
        print(f"File 2 Wins: {file2_wins} ({win_rate2:.2f}%)")

        if total_ties > 0:
            print(f"Ties: {total_ties} ({tie_rate:.2f}%)")
        print(f"(Based on {total_valid} total decisions)")

    for criterion in criteria:
        print_final_win_rate(criterion)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation and print win rates.")
    parser.add_argument("--file_1_path", type=str, default="Result/mini_cs/default_experiment/rkg_graph/node_only/Results/results.json", help="Path to the first file to evaluate.")
    parser.add_argument("--file_2_path", type=str, default="Result/mini_cs/node_merge_0.20_edge_0.65/rkg_graph/node_only/Results/results.json", help="Path to the second file to evaluate.")
    parser.add_argument("--output_dir", type=str, default="Result/cs/rkg_graph/evaluation_outputs", help="Path to the output directory.")
    parser.add_argument("--config_path", type=str, default="Option/Config2.yaml", help="Path to the config file.")
    args = parser.parse_args()
    FILE_1_PATH = args.file_1_path
    FILE_2_PATH = args.file_2_path
    OUTPUT_DIR = args.output_dir
    CONFIG_PATH = args.config_path
    asyncio.run(main())