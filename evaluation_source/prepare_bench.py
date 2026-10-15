import gzip
import json
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).parent

def nq_items():
    out = []
    with gzip.open(ROOT / "simplified-nq-dev.jsonl.gz", "rt", encoding="utf-8") as fh:
        for idx, line in enumerate(fh, 1):
            row = json.loads(line)
            tokens = row["document_text"].split()
            answers = []
            for ann in row.get("annotations", []):
                yn = ann.get("yes_no_answer")
                if yn and yn != "NONE":
                    answers.append(yn.lower())
                for span in ann.get("short_answers", []):
                    text = " ".join(tokens[span["start_token"]:span["end_token"]]).strip()
                    if text:
                        answers.append(text)
            answers = list(dict.fromkeys(answers))
            out.append({"id": idx, "source_id": str(row["example_id"]),
                        "question": row["question_text"], "answer": answers[0] if answers else "",
                        "answers": answers})
    return out

def hotpot_items():
    raw = json.loads((ROOT / "hotpot_dev_fullwiki_v1.json").read_text(encoding="utf-8"))
    return [{"id": idx, "source_id": str(row["_id"]), "question": row["question"],
             "answer": str(row["answer"]), "answers": [str(row["answer"])]}
            for idx, row in enumerate(raw, 1)]

for name, items in (("nq", nq_items()), ("hotpotqa", hotpot_items())):
    out = ROOT / f"benchmark_{name}_full.json"
    out.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    print(name, len(items), "empty_answers", sum(not x["answers"] for x in items), "->", out)

table = pq.read_table(ROOT / "searchr1_test.parquet", columns=["id", "question", "golden_answers", "data_source"])
data = table.to_pydict()
for name in ("nq", "hotpotqa"):
    items = []
    for row_id, question, answers, source in zip(data["id"], data["question"], data["golden_answers"], data["data_source"]):
        if source != name:
            continue
        answers = [str(x) for x in (answers or []) if str(x)]
        items.append({"id": str(row_id), "source_id": str(row_id), "question": str(question),
                      "answer": answers[0] if answers else "", "answers": answers})
    out = ROOT / f"benchmark_{name}_official.json"
    out.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    print("official", name, len(items), "empty_answers", sum(not x["answers"] for x in items), "->", out)

