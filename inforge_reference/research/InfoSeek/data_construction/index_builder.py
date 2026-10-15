import os
import json
from urllib.parse import quote


def build_title_index(root_dir):
    title_to_info = {}
    for subfolder in sorted(os.listdir(root_dir)):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for filename in sorted(os.listdir(subfolder_path)):
            if not filename.startswith("wiki_"):
                continue
            file_path = os.path.join(subfolder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_number, line in enumerate(f, start=1):  # Start counting from line 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            title = entry.get("title", "").strip()
                            if not title:
                                continue
                            key = quote(title, safe='')  # URL-encoded title for href matching
                            title_to_info[key] = {
                                "id": entry.get("id"),
                                "path": file_path,
                                "line_number": line_number,
                                "url": entry.get("url"),
                                "title": title
                            }
                        except json.JSONDecodeError as e:
                            print(f"[JSON Error] {file_path} line {line_number}: {e}")
            except Exception as e:
                print(f"[File Error] Failed to read file {file_path}: {e}")
    return title_to_info


# NOTE: Replace to the path of the unzipped wikidump
root_path = "path-to-your-unzipped-wikidump"
title_to_info = build_title_index(root_path)
# NOTE: Save the index to a specified path
output_path = "index-save-path"

print(f"Indexing complete. Total {len(title_to_info)} titles indexed.")

try:
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(title_to_info, f, ensure_ascii=False, indent=2)
    print(f"Index successfully saved to {output_path}")
except Exception as e:
    print(f"Failed to save index: {e}")

