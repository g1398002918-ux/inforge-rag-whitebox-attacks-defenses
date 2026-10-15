"""
Build FAISS index from a subset of wiki-25-512 corpus.
This creates a local search index suitable for an 8GB GPU.
"""
import json
import os
import sys
import argparse
import numpy as np
import faiss
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS index from wiki corpus subset")
    parser.add_argument("--wiki_path", type=str, required=True, help="Path to wiki-25-512.jsonl")
    parser.add_argument("--output_corpus", type=str, default="data/wiki_subset.jsonl", help="Output corpus file")
    parser.add_argument("--output_index", type=str, default="index/wiki_faiss.bin", help="Output FAISS index file")
    parser.add_argument("--num_docs", type=int, default=50000, help="Number of documents to index")
    parser.add_argument("--model_name", type=str, default="intfloat/e5-base-v2", help="Encoder model name")
    parser.add_argument("--batch_size", type=int, default=128, help="Encoding batch size")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Load subset of wiki corpus
    print(f"[1/4] Loading {args.num_docs} documents from wiki corpus...")
    subset = []
    with open(args.wiki_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= args.num_docs:
                break
            data = json.loads(line.strip())
            # Parse contents format: "title\ntext"
            contents = data.get('contents', '')
            parts = contents.split('\n', 1)
            title = parts[0] if parts else ''
            text = parts[1] if len(parts) > 1 else ''
            subset.append({
                'id': str(data.get('id', i)),
                'title': title,
                'text': text,
                'contents': contents,
            })
    print(f"   Loaded {len(subset)} documents")

    # 2. Save corpus file
    print(f"[2/4] Saving corpus to {args.output_corpus}...")
    os.makedirs(os.path.dirname(args.output_corpus), exist_ok=True)
    with open(args.output_corpus, 'w', encoding='utf-8') as f:
        for doc in subset:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    print(f"   Saved {len(subset)} documents")

    # 3. Load encoder model
    print(f"[3/4] Loading encoder model: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   Model on {device}")

    # 4. Encode all passages and build FAISS index
    print(f"[4/4] Encoding passages (batch_size={args.batch_size})...")
    all_embeddings = []

    for start in tqdm(range(0, len(subset), args.batch_size), desc="Encoding"):
        batch = subset[start:start + args.batch_size]
        # Format as passage for e5 model
        texts = [f"passage: {d['title']}. {d['text']}" for d in batch]

        inputs = tokenizer(
            texts,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)

        # Mean pooling
        attention_mask = inputs['attention_mask']
        last_hidden = output.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"   Embeddings shape: {all_embeddings.shape}")

    # Build FAISS index (inner product for normalized vectors)
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    # Save with GPU support if available
    os.makedirs(os.path.dirname(args.output_index), exist_ok=True)
    faiss.write_index(index, args.output_index)
    print(f"   Index saved to {args.output_index}")
    print(f"   Index size: {index.ntotal} vectors, dim={dim}")
    print(f"\n✅ Done! Use with:")
    print(f"   export CORPUS_PATH=\"{args.output_corpus}\"")
    print(f"   export INDEX_PATH=\"{args.output_index}\"")


if __name__ == "__main__":
    main()

