import json
import time
import traceback
from multiprocessing import Pool
from openai import OpenAI
import os

# NOTE: Input research tree JSONL file.
INPUT_FILE = "path-to-research-tree.jsonl"
# Output path
PASS_FILE = "./Passed.jsonl"
FAIL_FILE = "./Failed.jsonl"

# --- API Initialization Function (called by each subprocess) ---
def init_client():
    # NOTE: Initializes and returns an OpenAI client instance. Here we use deepseek as an example, please change to your own client and API key.
    return OpenAI(
        api_key="sk-xxxx", # Please ensure this is a valid API Key
        base_url="https://api.deepseek.com"
    )


# NOTE: You can change the promot to be more strict.
PROMPT_TEMPLATE = """
**Your Role:** You are a pragmatic AI Dataset Quality Analyst. Your task is to evaluate complex, multi-hop search-based questions to determine if they are good candidates for a dataset. **The dataset aims to train an LLM to conduct Deep Research.**

**Primary Objective:** Your goal is to identify high-quality questions that are best answered through a step-by-step investigation using a search tool, rather than relying solely on an LLM's internal knowledge.

### **Key Evaluation Guidelines**
Evaluate each question based on the following principles. A good question should generally align with these guidelines.

1.  **Multi-Step Nature:**
    *   The question should naturally break down into several logical steps to find the answer.
    *   Some of these steps should require an **external search**. It's acceptable if other steps involve reasoning, calculation, or synthesis based on the retrieved information.

2.  **Logical Connection:**
    *   The steps should be interconnected. Information from an earlier step is often needed to formulate the query for a later step. Avoid questions that are just a list of disconnected facts.

3.  **Clarity and Verifiability:**
    *   The question must be phrased clearly and be understandable.
    *   It should lead to a clear, fact-based answer. The answer can be a specific entity, a number, a date, or a concise summary/list. Questions that are purely subjective or opinion-based should be avoided.

4.  **Encourages a Multi-Step Path (Shortcut Test):**
    *   Consider if the question can be easily answered with a single, simple search query.
    *   A question is stronger if a multi-step process is the more **natural or reliable** way to arrive at the answer. It is acceptable if a highly complex, expert-level query could find a shortcut, as long as it isn't trivial to do so.

### **Your Output Format**
You must return your evaluation using the following **JSON structure strictly**:
```json
{{
  "Question": "{question}",
  "Analysis": "[Provide a concise justification for your decision, referencing the guidelines above.]",
  "Decision": "[Pass or Fail]"
}}
"""

# --- Multiprocessing Worker Function: Evaluate a single data item ---
def evaluate_data_item(index_data_tuple):
    """
    Evaluates a single data item via API call with comprehensive error handling.
    Any failure will result in a 'Fail' decision.
    """
    index, data_item = index_data_tuple
    
    try:
        question = data_item.get('root', {}).get('question')
        if not question:
            print(f"[{index+1}] ❌ FAIL: Question is missing or empty in data item.")
            return {'original_data': data_item, 'decision': 'Fail', 'error': 'MissingQuestion'}

        client = init_client()
        user_content = f"Please evaluate the following question based on the provided criteria:\n\n---\n{question}\n---\n\n{PROMPT_TEMPLATE}"
        
        print(f"[{index+1}] 🟡 API-Eval Start: \"{question[:60].strip()}...\"")
        start_time = time.time()

        # NOTE: Change to your own API client.
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": user_content}],
            stream=False,
            temperature=0.5,
        )

        result_text = response.choices[0].message.content.strip()
        elapsed = time.time() - start_time
        
        try:
            if result_text.startswith("```json"):
                result_text = result_text[7:-3].strip()
            evaluation_json = json.loads(result_text)
            # Normalize the Decision field, ensuring only "Pass" counts as a pass
            decision = evaluation_json.get("Decision", "Fail").strip().capitalize()
            if decision != "Pass":
                decision = "Fail"

            print(f"[{index+1}] ✅ API-Eval Done in {elapsed:.2f}s. Decision: {decision}")
            return {'original_data': data_item, 'decision': decision, 'error': None}
            
        except (json.JSONDecodeError, AttributeError):
            print(f"[{index+1}] ❌ FAIL: API response is not valid JSON. Elapsed: {elapsed:.2f}s.")
            print(f"[{index+1}] 📝 Raw Output:\n{result_text}\n{'-'*60}")
            return {'original_data': data_item, 'decision': 'Fail', 'error': 'InvalidJSONResponse'}

    except Exception as e:
        print(f"[{index+1}] ❌ FAIL: An unexpected error occurred during API evaluation.")
        traceback.print_exc()
        return {'original_data': data_item, 'decision': 'Fail', 'error': str(e)}

# --- Main Function ---
def main():
    main_start_time = time.time()
    
    items_to_evaluate_api = []
    parsing_fail_count = 0

    print(f"📖 Step 1: Reading and pre-filtering data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    data_item = json.loads(line)
                    items_to_evaluate_api.append(data_item)
                except json.JSONDecodeError:
                    print(f"⚠️ WARNING: Skipping malformed JSON on line {i+1}.")
                    parsing_fail_count += 1
    except FileNotFoundError:
        print(f"❌ FATAL ERROR: Input file not found at {INPUT_FILE}")
        return

    print("📊 Pre-filtering complete.")
    print(f"  - {parsing_fail_count} lines SKIPPED due to JSON parsing errors.")
    print(f"  - {len(items_to_evaluate_api)} items will be sent for API evaluation.")
            
    if not items_to_evaluate_api:
        print("\n✅ No items remaining for API evaluation. Process finished.")
        return

    # Step 2: Use multiprocessing to call the API for evaluation
    indexed_items = list(enumerate(items_to_evaluate_api))
    process_num = 100  # Number of processes to use
    print(f"\n🚀 Step 2: Starting parallel API evaluation for {len(indexed_items)} items using {process_num} processes...")

    with Pool(processes=process_num) as pool:
        results = pool.map(evaluate_data_item, indexed_items)

    print("\n✅ API evaluation complete.")
    print(f"💾 Step 3: Sorting and saving API results...")

    # Step 3: Sort and write data to Pass and Fail files based on API evaluation results
    api_pass_count = 0
    api_fail_count = 0
    with open(PASS_FILE, 'w', encoding='utf-8') as f_pass, \
         open(FAIL_FILE, 'a', encoding='utf-8') as f_fail:  # Append mode for the Fail file
        
        for result in results:
            if result and result.get('decision') == 'Pass':
                f_pass.write(json.dumps(result['original_data'], ensure_ascii=False) + '\n')
                api_pass_count += 1
            else:
                f_fail.write(json.dumps(result['original_data'], ensure_ascii=False) + '\n')
                api_fail_count += 1

    # --- Final Summary ---
    total_fail_count = api_fail_count
    total_processed = api_pass_count + total_fail_count
    
    print("\n--- FINAL SUMMARY ---")
    print(f"  - Total items processed: {total_processed}")
    print(f"  - Passed (API eval): {api_pass_count}")
    print(f"  - Failed (Total): {total_fail_count}")
    print(f"    - Failed by API eval / Error: {api_fail_count}")
    print(f"  - Skipped (Malformed JSON): {parsing_fail_count}")
    print("  --------------------")
    print(f"  -> Results PASSED saved to: {PASS_FILE}")
    print(f"  -> Results FAILED saved to: {FAIL_FILE}")

    main_end_time = time.time()
    total_elapsed = main_end_time - main_start_time
    minutes, seconds = divmod(total_elapsed, 60)
    print(f"\n🎉 Total execution time: {int(minutes)} minutes and {seconds:.2f} seconds.")


if __name__ == "__main__":
    main()

