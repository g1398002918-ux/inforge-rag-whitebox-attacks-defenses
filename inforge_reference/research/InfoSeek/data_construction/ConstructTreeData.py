import os
import sys
import json
import time
import random
import re
import ast
import threading
import concurrent.futures
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI
from transformers import AutoTokenizer
from tabulate import tabulate

from prompts import PLANNER_AGENT_PROMPT_TEMPLATE, BROWSER_AGENT_PROMPT_TEMPLATE


def generate_planner_prompt(current_question: str, tree_structure: Optional[Dict]) -> str:
    """Formats the prompt for the Planner Agent."""
    return PLANNER_AGENT_PROMPT_TEMPLATE.format(
        current_question=current_question,
        current_tree_structure=tree_structure
    )

def generate_browser_prompt(current_search_tree: Optional[Dict], entity_to_expand: str, action_to_execute: str, target_node_page_content: str) -> str:
    """Formats the prompt for the Browser Agent."""
    return BROWSER_AGENT_PROMPT_TEMPLATE.format(
        Current_Search_Tree=current_search_tree,
        Entity_To_Expand=entity_to_expand,
        Action_To_Execute=action_to_execute,
        Target_Node_Page_Content=target_node_page_content
    )

def extract_action_and_entity(output_string: str) -> Tuple[str, str]:
    """Extracts the action and entity from the Planner's output string."""
    action_pattern = r'\\Actionboxed\{(.*?)\}'
    entity_pattern = r'\\Entityboxed\{(.*?)\}'

    action_match = re.search(action_pattern, output_string)
    entity_match = re.search(entity_pattern, output_string)

    action = action_match.group(1) if action_match else ""
    entity = entity_match.group(1).strip('"') if entity_match else ""
    
    return action, entity

def extract_tree_and_href(browser_output: str) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Parses the Browser Agent's output to extract the updated search tree (JSON)
    and the href link of the next page to visit.
    """
    tree_data = None
    href_link = None
    
    # Extract the JSON tree structure
    try:
        start_marker = r'\tree_boxed{'
        start_index = browser_output.find(start_marker)
        if start_index != -1:
            content_start_index = start_index + len(start_marker)
            json_start_index = browser_output.find('{', content_start_index)
            if json_start_index != -1:
                brace_counter = 0
                json_end_index = -1
                for i in range(json_start_index, len(browser_output)):
                    char = browser_output[i]
                    if char == '{':
                        brace_counter += 1
                    elif char == '}':
                        brace_counter -= 1
                    if brace_counter == 0:
                        json_end_index = i
                        break
                
                if json_end_index != -1:
                    json_string = browser_output[json_start_index : json_end_index + 1]
                    tree_data = json.loads(json_string)
                else:
                    print(f"Warning (Thread {threading.get_ident()}): Found 'tree_boxed' start but no matching end brace.")
            else:
                 print(f"Warning (Thread {threading.get_ident()}): Found 'tree_boxed' but no JSON object start '{{' within.")
    except json.JSONDecodeError as e:
        print(f"JSON parsing failed (Thread {threading.get_ident()}): {e}")
        tree_data = None
    except Exception as e:
        print(f"An unexpected error occurred during 'tree_boxed' extraction (Thread {threading.get_ident()}): {e}")
        tree_data = None

    # Extract the href link
    href_match = re.search(r'\\href_boxed\{.*?href="([^"]+)".*?\}', browser_output, re.DOTALL)
    if href_match:
        href_link = href_match.group(1)
        
    return tree_data, href_link

def read_dict_from_line(file_path: str, line_number: int) -> Optional[Dict]:
    """Reads a specific line from a file and parses it as a Python dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i + 1 == line_number:
                    try:
                        data_dict = ast.literal_eval(line.strip())
                        if isinstance(data_dict, dict):
                            return data_dict
                        else:
                            print(f"Error (Thread {threading.get_ident()}): Content on line {line_number} is not a dictionary.")
                            return None
                    except (ValueError, SyntaxError) as e:
                        print(f"Error (Thread {threading.get_ident()}): Could not parse line {line_number}. Error: {e}")
                        return None
            print(f"Error (Thread {threading.get_ident()}): Line number {line_number} is out of file bounds.")
            return None
    except FileNotFoundError:
        print(f"Error (Thread {threading.get_ident()}): File not found -> {file_path}")
        return None

def process_and_truncate_text(result_dict: Dict[str, Any], tokenizer: Any, max_length: int = 1024) -> Dict[str, Any]:
    """Truncates the 'text' field in a dictionary to a maximum token length."""
    result_dict_truncated = result_dict.copy()
    if 'text' not in result_dict_truncated or not isinstance(result_dict_truncated['text'], str):
        print(f"Warning (Thread {threading.get_ident()}): Input dict has no 'text' key or it's not a string. Returning a copy.")
        return result_dict_truncated
        
    token_ids = tokenizer.encode(result_dict_truncated['text'])
    if len(token_ids) > max_length:
        truncated_token_ids = token_ids[:max_length]
        new_text = tokenizer.decode(truncated_token_ids, skip_special_tokens=True)
        result_dict_truncated['text'] = new_text
    
    return result_dict_truncated

def get_new_wiki_page(tree: Dict, target_entity: str, title_to_info: Dict, tokenizer: Any) -> Optional[str]:
    """
    Retrieves and processes the content of a Wikipedia page for a target entity
    based on the href found in the search tree.
    """
    def dfs(node: Dict) -> Optional[str]:
        if node.get("entity") == target_entity:
            return node.get("href")
        for child in node.get("children", []):
            result = dfs(child)
            if result:
                return result
        return None

    current_href = dfs(tree.get("root", {}))
    
    if not current_href:
        print(f"Could not find href for entity '{target_entity}' in the tree.")
        return None

    try:
        page_info = title_to_info.get(current_href)
        if not page_info:
            print(f"Failed to find info for href '{current_href}' in the index.")
            return None
        
        result_dict = read_dict_from_line(page_info['path'], page_info['line_number'])
        if not result_dict:
            return None

        result_dict_truncated = process_and_truncate_text(result_dict, tokenizer)
        target_node_page_content = result_dict_truncated['title'] + result_dict_truncated['text']
        return target_node_page_content
    except Exception as e:
        print(f"Failed to get content for target node '{target_entity}': {e}")
        return None



def process_single_item(item_id: int, title_to_info: Dict, keys_list: List, tokenizer: Any, agent: OpenAI) -> Optional[Dict]:
    """
    Processes the construction of a single data item. This function is executed
    in a separate thread.

    It starts with a random Wikipedia page and iteratively uses Planner and Browser
    agents to build a multi-hop reasoning path (search tree).

    Args:
        item_id: A unique identifier for the task/thread.
        title_to_info: A dictionary mapping Wikipedia page titles to their location.
        keys_list: A list of all Wikipedia page titles for random sampling.
        tokenizer: The tokenizer for calculating text length.
        agent: The OpenAI client instance for making API calls.

    Returns:
        A dictionary representing the final constructed search tree if successful,
        otherwise None.
    """
    print(f"[Task {item_id}] Starting processing...")
    
    # Initialize by selecting a random root Wikipedia page.
    valid_initial_page = False
    init_result_dict_truncated = None
    attempts = 0
    while not valid_initial_page and attempts < 10:
        random_key = random.choice(keys_list)
        popped_value = title_to_info.get(random_key)
        if not popped_value:
            attempts += 1
            continue
        
        result_dict = read_dict_from_line(popped_value['path'], popped_value['line_number'])
        if result_dict and len(result_dict.get('text', '')) > 10:
            valid_initial_page = True
            init_result_dict_truncated = process_and_truncate_text(result_dict, tokenizer)
        attempts += 1
    
    if not init_result_dict_truncated:
        print(f"[Task {item_id}] Failed to initialize a valid root wiki page. Task aborted.")
        return None

    root_wiki_content = init_result_dict_truncated['title'] + ' ' + init_result_dict_truncated['text']

    # Start the iterative construction process.
    current_action, current_search_tree, iter_num = None, None, 0
    max_iter_num = 5
    all_tree_states = []
    
    while current_action != "Action 4" and iter_num < max_iter_num:
        iter_num += 1
        print(f"[Task {item_id}] --- Iteration {iter_num} ---")
        
        # --- Planner Agent ---
        if current_search_tree is None:
            # On the first iteration, there is no tree yet.
            if iter_num > 2:
                print(f"[Task {item_id}] Failed to generate a tree from the root page. Aborting early at iter {iter_num}.")
                break
            current_question = "No question yet. Let's use Action 1 to start the loop!"
        else:
            current_question = current_search_tree['root']['question']
        
        planner_prompt = generate_planner_prompt(current_question, current_search_tree)
        try:
            response = agent.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": planner_prompt}],
                temperature=0.9,
            )
            planner_output = response.choices[0].message.content.strip()
            print(f"[Task {item_id}] Planner Agent Output: {planner_output}")
            current_action, current_entity = extract_action_and_entity(planner_output)
        except Exception as e:
            print(f"[Task {item_id}] Planner Agent API call failed: {e}. Aborting at iter {iter_num}.")
            break

        # --- Prepare Browser Agent Input ---
        if current_search_tree is None:
            # First iteration uses the initial root page content.
            current_entity = init_result_dict_truncated['title']
            target_node_page_content = root_wiki_content
        else:
            print(f"Current Action: {current_action}")
            if current_action == "Action 4" or '4' in current_action:
                print(f"Current Entity: {current_entity}")
                if "Quit" in current_entity:
                    print(f"[Task {item_id}] Construction failed: Planner deemed data quality low. Aborting at iter {iter_num}.")
                    current_search_tree = None # Signal failure
                    break
                print(f"[Task {item_id}] Iteration limit reached or planner finished. Terminating construction at iter {iter_num}.")
                break
            
            target_node_page_content = get_new_wiki_page(current_search_tree, current_entity, title_to_info, tokenizer)
            if target_node_page_content is None:
                print(f"[Task {item_id}] Failed to retrieve wiki page content. Terminating construction at iter {iter_num}.")
                break
        
        # --- Browser Agent ---
        browser_prompt = generate_browser_prompt(current_search_tree, current_entity, current_action, target_node_page_content)
        try:
            response = agent.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": browser_prompt}],
                temperature=0.8,
            )
            browser_output = response.choices[0].message.content.strip()
            print(f"[Task {item_id}] Browser Agent Output: {browser_output}")
            new_search_tree, new_href = extract_tree_and_href(browser_output)
        except Exception as e:
            print(f"[Task {item_id}] Browser Agent API call failed: {e}. Aborting at iter {iter_num}.")
            break
        
        # --- Update State ---
        # Verify the new page content before updating the tree.
        if new_href and new_href in title_to_info:
            value = title_to_info[new_href]
            result_dict = read_dict_from_line(value['path'], value['line_number'])
            # Check if the linked page has substantial content.
            if result_dict and len(result_dict.get('text', '')) > 100:
                if new_search_tree:
                    current_search_tree = new_search_tree
                    all_tree_states.append(new_search_tree['root']) # Save intermediate states
                    print(f"[Task {item_id}] Successfully updated the search tree.")
                else:
                    print(f"[Task {item_id}] Failed to parse the new search tree. Terminating at iter {iter_num}.")
                    break
            else:
                print(f"[Task {item_id}] Target Wiki page is invalid or too short. Terminating at iter {iter_num}.")
                break
        else:
            print(f"[Task {item_id}] Link '{new_href}' not found in wiki dump. Terminating at iter {iter_num}.")
            break
            
    if current_search_tree:
        current_search_tree['all_tree_states'] = all_tree_states
        current_search_tree['hop_num'] = len(all_tree_states)
        print(f"[Task {item_id}] Task completed successfully. Final data has {current_search_tree['hop_num']} hops.")
        return current_search_tree
    else:
        print(f"[Task {item_id}] Task failed to produce valid data.")
        return None


# --- Main Execution Logic ---
if __name__ == "__main__":
    main_start_time = time.time()
    print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")

    # --- Global Configuration ---
    NUM_TASKS = 16000
    MAX_WORKERS = 100
    INPUT_WIKI_INDEX_PATH = "/path/to/your/Wikipedia/WikiIndex.json"
    TOKENIZER_PATH = "/path/to/your/Qwen2.5-7B-Instruct"
    API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_api_key_here")
    BASE_URL = "https://api.deepseek.com"
    OUTPUT_PATH = "/path/to/your/ConstructedTreeData/generated_data.jsonl"


    print("--- Prompt Templates ---")
    print(f"Planner Agent Prompt Preview: {PLANNER_AGENT_PROMPT_TEMPLATE[:200]}...\n")
    print(f"Browser Agent Prompt Preview: {BROWSER_AGENT_PROMPT_TEMPLATE[:200]}...\n")

    # --- 1. Initialize Shared Resources ---
    print("Loading shared resources...")
    start_time = time.time()
    try:
        with open(INPUT_WIKI_INDEX_PATH, "r", encoding="utf-8") as f:
            title_to_info = json.load(f)
        keys_list = list(title_to_info.keys())
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        agent = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        print(f"Successfully loaded index with {len(title_to_info)} records. Time taken: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Initialization failed: {e}")
        sys.exit(1)

    # --- 2. Execute Tasks using a Thread Pool ---
    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {executor.submit(process_single_item, i, title_to_info, keys_list, tokenizer, agent): i for i in range(NUM_TASKS)}
        
        print(f"\nSubmitted {NUM_TASKS} tasks. Processing with {MAX_WORKERS} worker threads...\n")

        for future in concurrent.futures.as_completed(future_to_id):
            item_id = future_to_id[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    print(f"--- Main Thread: Collected a valid result from Task {item_id}. Total results: {len(all_results)} ---")
            except Exception as exc:
                print(f'[Task {item_id}] generated an exception during execution: {exc}')

    print(f"\nAll tasks completed. Successfully generated {len(all_results)} data items.")

    # --- 3. Analyze and Save Results ---
    if not all_results:
        print("No data was generated. Exiting.")
        sys.exit(0)

    # --- Statistics on Hop Distribution ---
    hop_counts = Counter(item.get('hop_num', 0) for item in all_results)
    total_generated = len(all_results)
    
    # Filter for valid data (e.g., hop_num >= 2)
    valid_data = [item for item in all_results if item.get("hop_num", 0) >= 2]
    valid_count = len(valid_data)
    success_rate = valid_count / NUM_TASKS if NUM_TASKS > 0 else 0

    print("\n--- Generation Statistics ---")
    summary_table = [
        ["Total Attempts", NUM_TASKS, f"{NUM_TASKS / NUM_TASKS:.2%}"],
        ["Successful Generations", total_generated, f"{total_generated / NUM_TASKS:.2%}"],
        ["Valid Data (Hops >= 2)", valid_count, f"{success_rate:.2%}"]
    ]
    print(tabulate(summary_table, headers=["Metric", "Count", "Rate"], tablefmt="fancy_grid"))

    # Full hop distribution table
    print("\n--- Full Hop Distribution ---")
    hop_table = [[hop, count, f"{count / total_generated:.2%}"] for hop, count in sorted(hop_counts.items())]
    print(tabulate(hop_table, headers=["Hop Count", "Frequency", "Percentage"], tablefmt="grid"))

    # --- Save Valid Data ---
    if valid_data:
        print(f"\nSaving {valid_count} valid data items (hop_num >= 2) to {OUTPUT_PATH}...")
        try:
            os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                for item in valid_data:
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + "\n")
            print("Data successfully saved!")
        except Exception as e:
            print(f"Failed to save data: {e}")
    else:
        print("No valid data with hop_num >= 2 to save.")


    main_end_time = time.time()
    total_duration_minutes = (main_end_time - main_start_time) / 60
    print("\n" + "="*50)
    print(f"Total script execution time: {total_duration_minutes:.2f} minutes")
    print("="*50)

