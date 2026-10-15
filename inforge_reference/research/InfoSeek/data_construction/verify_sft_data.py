import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# NOTE: Initializes and returns an OpenAI client instance. Here we use deepseek as an example, please change to your own client and API key.
client = OpenAI(
    api_key="sk-xxxx",
    base_url="https://api.deepseek.com"
)

# Input and output file paths
input_file = "path-to-trajectory-sft-data.jsonl"
output_file = "./verified_sft_data.jsonl"

# Prompt template for filtering trajectories
input_prompt_template = """
You are tasked with evaluating whether a trajectory is high-quality training data for Supervised Fine-Tuning (SFT) of an LLM as a Deep Research Agent. Assess it under the following strict criteria:

1. **Step-by-Step Reasoning**
- The agent must break the question into sub-problems and reason progressively toward the answer.
- Direct shortcuts to the answer without multi-hop reasoning are not allowed.

2. **Search Quality**
- Searches must be justified by prior reasoning, not speculative guesswork.
- Trajectories where the question is trivially answerable by a single query (too easy, no real reasoning required) must be rejected.

3. **Output Format**
- First, provide a concise evaluation of whether the trajectory meets the above criteria.
- Then, on a new line, output only the verdict in the form: \\boxed{{Pass}} or \\boxed{{Failed}}.

Trajectory:
{text}
"""


def evaluate_trajectory(conversation):
    """
    Calls the API to evaluate a single data item (trajectory).
    """
    try:
        text = ""
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            text += f"{role.upper()}:\n{content}\n\n"

        # Randomly decide whether to log the input (0.5% chance) for monitoring
        if random.random() < 0.005:
            print("\n" + "="*60)
            print("LOGGING SAMPLE (0.5% probability)")
            print("Input:")
            print(text[:300] + "..." if len(text) > 300 else text)
            print("-" * 60)

        # NOTE: Change to your own API client.
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": input_prompt_template.format(text=text)}],
            stream=False,
            temperature=0.5,
        )

        output = response.choices[0].message.content.strip()

        if random.random() < 0.01:
            print("Full API Output:")
            print(output)
            print("="*60)

        if "\\boxed{Pass}" in output:
            return True  # Keep the item
        else:
            return False  # Filter out the item
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        return False


def main():
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    # data = data[:100]  # Uncomment for testing with a smaller subset of data
    results = []

    with ThreadPoolExecutor(max_workers=100) as executor:
        future_to_item = {executor.submit(evaluate_trajectory, item["conversation"]): item for item in data}

        for future in tqdm(as_completed(future_to_item), total=len(data), desc="Filtering Data"):
            item = future_to_item[future]
            try:
                keep = future.result()
                if keep:
                    results.append(item)
            except Exception as e:
                print(f"An exception occurred in a future: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Finished. Kept {len(results)} / {len(data)} high-quality data items.")


if __name__ == "__main__":
    main()

