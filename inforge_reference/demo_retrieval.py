#!/usr/bin/env python3
"""
InfoSeek Retrieval System — Demo Script
用于向老师演示 InfoSeek 论文复现成果

调用本地检索服务 (http://localhost:28706) 展示文档检索效果
"""

import requests
import json
import sys
import time
from typing import List, Dict

SERVER_URL = "http://localhost:28706"
VERBOSE = True


def check_health() -> dict:
    """检查检索服务状态"""
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json()
        return {"status": "error", "detail": f"HTTP {resp.status_code}"}
    except requests.ConnectionError:
        return {"status": "unreachable", "detail": "服务未启动，请先运行 start_server.bat"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def retrieve(queries: List[str], topk: int = 5, scores: bool = True) -> dict:
    """调用批量检索接口"""
    resp = requests.post(
        f"{SERVER_URL}/retrieve",
        json={"queries": queries, "topk": topk, "return_scores": scores},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def print_separator(title: str = ""):
    """打印分隔线"""
    width = 72
    if title:
        side = (width - len(title) - 2) // 2
        print(f"{'─' * side} {title} {'─' * side}")
    else:
        print("─" * width)


def print_result(query: str, results: list, idx: int = None):
    """格式化打印单条查询的检索结果"""
    if idx is not None:
        print(f"\n🔍 Query {idx}: {query}")
    else:
        print(f"\n🔍 Query: {query}")
    print(f"   {'Rank':<5} {'Score':<8} {'Title':<40} {'Text (前80字)'}")
    print(f"   {'─'*4:<5} {'─'*7:<8} {'─'*39:<40} {'─'*20}")

    for i, item in enumerate(results):
        if isinstance(item, dict) and "document" in item:
            doc = item["document"]
            score = item.get("score", 0.0)
        else:
            doc = item
            score = 0.0

        title = doc.get("title", "N/A")[:38]
        text = doc.get("text", "")[:80].replace("\n", " ")
        print(f"   {i+1:<5} {score:<8.4f} {title:<40} {text}")


# ── 预设演示查询 ──────────────────────────────────────────────

DEMO_QUERIES = [
    # 第一组：基础事实类
    [
        "What is reinforcement learning?",
        "Who wrote Hamlet?",
        "Capital of France",
    ],
    # 第二组：深度学习/AI 类
    [
        "How does the Transformer architecture work?",
        "What is a neural network?",
        "Information foraging theory",
    ],
    # 第三组：交叉领域（展示覆盖面）
    [
        "Global warming climate change evidence",
        "History of artificial intelligence",
        "How does a search engine retrieve documents?",
    ],
]


def run_demo():
    """运行完整演示流程"""
    print()
    print("╔" + "═" * 70 + "╗")
    print("║" + "   🔍 InfoSeek Retrieval System — 本地部署复现演示".center(64) + "║")
    print("║" + "   NeurIPS 2025 InForage / InfoSeek 论文复现".center(62) + "║")
    print("╚" + "═" * 70 + "╝")
    print()

    # 1. 健康检查
    print_separator("系统状态")
    health = check_health()
    if health.get("status") != "ok":
        print(f"❌ 服务异常: {health.get('detail', 'Unknown error')}")
        return False

    print(f"  ✅ 服务正常")
    print(f"  📦 模型: {health.get('model', 'N/A')}")
    print(f"  📊 索引规模: {health.get('index_size', 0):,} 篇文档")
    print()

    # 2. 系统信息
    print_separator("技术栈")
    print(f"  🖥️  GPU:     NVIDIA RTX 5070 Laptop (8GB, Blackwell sm_120)")
    print(f"  🔥  PyTorch: 2.12.0.dev+cu128 (nightly)")
    print(f"  🧠  模型:    intfloat/e5-base-v2 (768维稠密向量)")
    print(f"  📚  语料:    wiki-25-512 子集 (20万篇 Wikipedia 文档)")
    print(f"  ⚡  索引:    FAISS IndexFlatIP (内积检索)")
    print(f"  🌐  接口:    FastAPI + Uvicorn (RESTful)")
    print()

    # 3. 逐组运行检索演示
    total_queries = 0
    total_time = 0.0

    for group_idx, queries in enumerate(DEMO_QUERIES, 1):
        print_separator(f"演示组 {group_idx}")

        t0 = time.time()
        try:
            data = retrieve(queries, topk=5, scores=True)
            elapsed = time.time() - t0
            total_time += elapsed
            total_queries += len(queries)

            for i, results in enumerate(data["result"]):
                print_result(queries[i], results, idx=i + 1)

            print(f"\n  ⏱️ 本组 {len(queries)} 条查询耗时: {elapsed:.2f}s")
        except Exception as e:
            print(f"  ❌ 检索失败: {e}")
            return False

        print()

    # 4. 性能总结
    print_separator("性能总结")
    print(f"  📊  总查询数:  {total_queries}")
    print(f"  ⏱️   总耗时:    {total_time:.2f}s")
    print(f"  🚀  平均延迟:  {total_time/total_queries*1000:.0f}ms/query")
    print(f"  📦  索引大小:  {health.get('index_size', 0):,} 篇 × 768维 = "
          f"{health.get('index_size', 0) * 768 * 4 / (1024**2):.0f} MB 显存")
    print()

    print("✅ 演示完成！检索服务运行正常。")
    return True


def run_interactive():
    """交互模式：用户输入查询，实时返回结果"""
    print()
    print("╔" + "═" * 60 + "╗")
    print("║" + "   InfoSeek 交互式检索".center(52) + "║")
    print("║" + "   输入 'quit' 或 'exit' 退出".center(48) + "║")
    print("╚" + "═" * 60 + "╝")
    print()

    # 先检查服务
    health = check_health()
    if health.get("status") != "ok":
        print(f"❌ 服务不可用: {health.get('detail')}")
        print("请先运行: start_server.bat")
        return

    print(f"✅ 服务已连接 ({health.get('index_size', 0):,} 篇文档)")
    print()

    while True:
        try:
            query = input("🔍 请输入查询: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 退出")
            break

        if query.lower() in ("quit", "exit", "q", "退出"):
            print("👋 退出")
            break

        if not query:
            continue

        t0 = time.time()
        try:
            data = retrieve([query], topk=5, scores=True)
            elapsed = time.time() - t0
            print_result(query, data["result"][0])
            print(f"  ⏱️ {elapsed:.3f}s\n")
        except Exception as e:
            print(f"  ❌ 检索失败: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("-i", "--interactive"):
        run_interactive()
    elif len(sys.argv) > 1 and sys.argv[1] in ("-c", "--check"):
        health = check_health()
        print(json.dumps(health, indent=2, ensure_ascii=False))
    elif len(sys.argv) > 1 and sys.argv[1] in ("-q", "--query"):
        if len(sys.argv) < 3:
            print("用法: python demo_retrieval.py -q 'your query here'")
            sys.exit(1)
        health = check_health()
        if health.get("status") != "ok":
            print(f"❌ 服务不可用: {health.get('detail')}")
            sys.exit(1)
        data = retrieve([sys.argv[2]], topk=5, scores=True)
        print_result(sys.argv[2], data["result"][0])
    else:
        run_demo()

