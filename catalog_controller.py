from __future__ import annotations
"""
catalog_controller.py — 编目流程控制器（状态机）

状态: INIT → CLUSTER → ASSESS → {ACCEPT|SPLIT|BIN} → CHECK_STOP → EXPORT

用法:
    python3 catalog_controller.py                     # 从头跑
    python3 catalog_controller.py --resume             # 从 state 恢复
    python3 catalog_controller.py --stage EXPORT       # 只导出
"""

import ast
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from catalog_state import CatalogState
from rule_scorer import RuleScorer
from llm_judge import LLMJudge
from split_engine import split_community

# 输出目录
OUTPUT_DIR = Path(os.environ.get("CATALOG_OUTPUT_DIR",
    "/Users/yuwan/code/news-clustering/output_catalog"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Leiden 输出
LEIDEN_DIR = Path(os.environ.get("CATALOG_LEIDEN_DIR",
    "/Users/yuwan/code/news-clustering/output_exp_i"))

# 源数据
SOURCE_CSV = os.environ.get("CATALOG_SOURCE_CSV",
    "/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")

# 全局数据（懒加载）
_GLOBAL: dict = {}


def _load_globals():
    """加载并缓存 headline 列表 + embeddings + 社区分配"""
    if _GLOBAL:
        return _GLOBAL

    print("  [加载] headline 列表...")
    df = pd.read_csv(SOURCE_CSV)
    title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
    _GLOBAL['headlines'] = df[title_col].astype(str).tolist()

    print("  [加载] embeddings...")
    emb_path = LEIDEN_DIR / "embeddings.npy"
    if emb_path.exists():
        _GLOBAL['embeddings'] = np.load(emb_path)
        print(f"    {_GLOBAL['embeddings'].shape}")
    else:
        _GLOBAL['embeddings'] = None
        print("    (无 embedding 文件，跳过样本选取)")

    print("  [加载] 社区分配...")
    comm_path = LEIDEN_DIR / "headlines_with_topics.csv"
    if comm_path.exists():
        df_comm = pd.read_csv(comm_path)
        topic_col = [c for c in df_comm.columns if "topic" in c.lower()][0]
        _GLOBAL['topic_ids'] = df_comm[topic_col].values
    else:
        _GLOBAL['topic_ids'] = None

    return _GLOBAL


# ==================== 共享工具函数 ====================


def _compute_median_dist(embeddings: np.ndarray, n_samples: int = 2000) -> float:
    """在全量 embedding 中采样计算中位余弦距离（用于 Rule 5 基线）"""
    n = embeddings.shape[0]
    rng = np.random.default_rng(42)
    idx_a = rng.integers(0, n, size=n_samples)
    idx_b = rng.integers(0, n, size=n_samples)
    # 确保不是同一对
    same = idx_a == idx_b
    if same.any():
        idx_b[same] = (idx_b[same] + 1) % n
    sims = (embeddings[idx_a] * embeddings[idx_b]).sum(axis=1)
    sims = np.clip(sims, -1.0, 1.0)
    dists = 1.0 - sims
    return float(np.median(dists))


def _resolve_indices(node, topic_ids):
    """解析节点对应的全局索引数组。

    子社区节点（有 _sub_indices）：直接返回。
    原始社区节点（v0_*）：通过 topic_id 匹配。
    返回 np.ndarray 或 None（解析失败）。
    """
    # 子社区节点
    sub_indices = node.get("_sub_indices")
    if sub_indices:
        return np.array(sub_indices, dtype=int)

    # 原始社区节点
    if topic_ids is None:
        return None
    node_idx = int(node["id"].split("_c")[1]) if "_c" in node["id"] else 0
    unique_ids = np.unique(topic_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    sorted_ids = sorted(unique_ids)
    if node_idx >= len(sorted_ids):
        return None
    cid = sorted_ids[node_idx]
    mask = topic_ids == cid
    return np.where(mask)[0]


def _get_samples(node, headlines, embeddings, topic_ids):
    """取社区的代表标题（centroid 最近 10 条）和边界标题（centroid 最远 5 条）"""
    if embeddings is None:
        return [], []

    indices = _resolve_indices(node, topic_ids)
    if indices is None or len(indices) == 0:
        return [], []

    # 社区 centroid
    community_embs = embeddings[indices]
    centroid = community_embs.mean(axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 1e-10:
        sims = community_embs @ centroid
    else:
        sims = np.ones(community_embs.shape[0])
    order = np.argsort(-sims)

    top_5 = [headlines[indices[i]] for i in order[:5]]
    bottom_3 = [headlines[indices[i]] for i in order[-3:]]

    return top_5, bottom_3


# LLM judge 实例
llm_judge = LLMJudge(cache_path=OUTPUT_DIR / "llm_cache.json")


# ==================== 阶段函数 ====================


def stage_init(state: CatalogState):
    """INIT: 初始化状态"""
    print("\n[INIT] 初始化...")
    df = pd.read_csv(SOURCE_CSV)
    total = len(df)
    state.init_new(total_headlines=total, max_rounds=3, round=0)
    print(f"  总标题数: {total}")
    print(f"  state 文件: {state.state_path}")


def stage_cluster(state: CatalogState):
    """CLUSTER: 加载 Leiden 输出 → 写入 state"""
    print("\n[CLUSTER] 加载聚类结果...")

    # 从 Leiden 输出加载
    leiden_dir = Path("/Users/yuwan/code/news-clustering/output_exp_i")
    if not leiden_dir.exists():
        print("  ✗ Leiden 输出未找到，先运行 experiment_i.py")
        return

    # 加载社区分配
    comm_file = leiden_dir / "headlines_with_topics.csv"
    if not comm_file.exists():
        print(f"  ✗ 未找到 {comm_file}")
        return

    df_comm = pd.read_csv(comm_file)
    topic_col = [c for c in df_comm.columns if "topic" in c.lower()][0]

    # 读 topic_info 获取关键词
    info_file = leiden_dir / "topic_info.csv"
    df_info = pd.read_csv(info_file)
    info_col = [c for c in df_info.columns if "topic" in c.lower()][0]  # Topic 或 topic_id
    kw_col = [c for c in df_info.columns if "keyword" in c.lower() or "representation" in c.lower()][0]

    # 构建关键词映射
    kw_map = {}
    for _, row in df_info.iterrows():
        tid = row[info_col]
        kw = row[kw_col]
        if isinstance(kw, str):
            # 可能是 list 字符串
            if kw.startswith("["):
                try:
                    kw = ", ".join(ast.literal_eval(kw)[:10])
                except:
                    pass
        kw_map[int(tid)] = str(kw)

    # 社区统计
    topic_ids = df_comm[topic_col].values
    clusters = np.unique(topic_ids)
    clusters = clusters[clusters >= 0]  # 排除 -1（如果 Leiden 有离群）

    print(f"  加载 {len(clusters)} 个社区")

    added = 0
    for i, cid in enumerate(sorted(clusters)):
        mask = topic_ids == cid
        size = int(mask.sum())
        keywords = kw_map.get(int(cid), "")
        node = {
            "id": f"v0_c{i}",
            "parent": None,
            "depth": 0,
            "size": size,
            "keywords": keywords,
            "mean_pairwise_dist": None,
            "rule_grade": None,
            "rule_signals": {},
            "llm_grade": None,
            "llm_name": None,
            "llm_action": None,
            "final_grade": None,
            "action": None,
            "children": [],
            "status": "pending",
        }
        state.add_node(node)
        added += 1

    print(f"  新增 {added} 个社区节点")

    # ---- 社区自动分组：小社区+语义相近 → 挂到大根下面 ----
    try:
        g = _load_globals()
        emb = g.get('embeddings')
        topic_ids = g.get('topic_ids')
        if emb is not None and topic_ids is not None:
            nodes = state.state["nodes"]
            # 按 id 排序确保稳定
            v0_nodes = sorted([n for n in nodes if n["depth"] == 0], key=lambda x: x["id"])
            centroids = []
            for n in v0_nodes:
                idx = _resolve_indices(n, topic_ids)
                if idx is None or len(idx) == 0:
                    centroids.append(np.zeros(emb.shape[1]))
                else:
                    centroids.append(emb[idx].mean(axis=0))
            centroids = np.array(centroids)
            norms = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            centroids = centroids / norms
            sims = centroids @ centroids.T

            # 贪心分组：从大到小，仅当小社区的 top-1 相似是大根
            sorted_idx = sorted(range(len(v0_nodes)), key=lambda i: -v0_nodes[i]["size"])
            grouped = set()
            for i in sorted_idx:
                parent_node = v0_nodes[i]
                if i in grouped:
                    continue
                # 找每个小社区的最相似大根
                candidates = []
                for j in range(len(v0_nodes)):
                    if i == j or j in grouped:
                        continue
                    child = v0_nodes[j]
                    sz_ratio = child["size"] / parent_node["size"]
                    if sz_ratio < 0.35 and sims[i][j] > 0.55:
                        # 确认小社区最相似的就是这个 parent
                        best = max((sims[k][j], k) for k in range(len(v0_nodes)) if k != j and k not in grouped)
                        if best[1] == i:
                            candidates.append((j, sims[i][j]))
                candidates.sort(key=lambda x: -x[1])
                for j, _ in candidates:
                    child = v0_nodes[j]
                    child["parent"] = parent_node["id"]
                    child["depth"] = 1
                    parent_node.setdefault("children", []).append(child["id"])
                    grouped.add(j)
                    print(f"  分组: {child['id']}「{child.get('llm_name','')}」({child['size']}) → 挂到 {parent_node['id']}「{parent_node.get('llm_name','')}」({parent_node['size']})  sim={sims[i][j]:.3f}")
            if grouped:
                state.save()
                print(f"  分组完成: {len(grouped)} 个社区重新链接")
    except Exception as e:
        print(f"  [分组跳过] {e}")


def stage_assess(state: CatalogState):
    """ASSESS: 规则打分 + LLM（待处理节点）"""
    print(f"\n[ASSESS] 评估节点 (round={state.current_round()})...")

    pending = state.get_pending_nodes()
    if not pending:
        print("  没有待处理节点")
        return

    total = state.state["total_headlines"]

    # 加载 embeddings 计算基线距离（仅计算一次）
    g = _load_globals()
    median_dist = None
    if g.get('embeddings') is not None:
        median_dist = _compute_median_dist(g['embeddings'])
        print(f"  [规则] 全量中位余弦距离: {median_dist:.4f}")

    scorer = RuleScorer(total_headlines=total, median_all_dist=median_dist)

    llm_count = 0
    for node in pending:
        # 深度≥2 的节点直接接受（不再细分）
        if node.get("depth", 0) >= 2:
            kw = node.get("keywords", "")
            name = "、".join(kw.split(", ")[:3]) if kw else "杂项"
            state.update_node(node["id"], {
                "final_grade": "B",
                "llm_grade": "B",
                "llm_name": name,
                "action": "accept",
                "status": "accepted",
            })
            print(f"  {node['id']}: depth≥2 → auto accept「{name}」({node['size']}条)")
            continue

        # 规则打分
        grade, signals, _ = scorer.score(node)
        state.update_node(node["id"], {
            "rule_grade": grade,
            "rule_signals": signals,
        })

        if grade is not None:
            # 规则已经确定 → 直接归档
            if grade in ("D-meta", "D-fragment"):
                action = "bin_meta" if grade == "D-meta" else "bin_fragment"
                status = "binned"
                print(f"  {node['id']}: 规则→{grade} ({action})")
            elif grade == "C":
                action = "split"
                status = "split"
                print(f"  {node['id']}: 规则→C (split)")
            else:
                action = "accept"
                status = "accepted"
                print(f"  {node['id']}: 规则→{grade} (accept)")

            state.update_node(node["id"], {
                "final_grade": grade,
                "action": action,
                "status": status,
            })
        else:
            # 规则不确定 → 走 LLM
            llm_count += 1
            top_samples, boundary_samples = _get_samples(
                node, g['headlines'], g['embeddings'], g['topic_ids']
            )

            result = llm_judge.judge(
                node_id=node["id"],
                size=node["size"],
                keywords=node.get("keywords", ""),
                top_samples=top_samples,
                boundary_samples=boundary_samples,
            )

            if "error" in result:
                print(f"  {node['id']}: LLM 失败 → B/accept")
                state.update_node(node["id"], {
                    "final_grade": "B",
                    "action": "accept",
                    "status": "accepted",
                })
                continue

            grade = result.get("grade", "B")
            name = result.get("name_zh", "")
            llm_action = result.get("action", "keep")

            if grade in ("A", "B"):
                action = "accept"
                status = "accepted"
            elif grade == "C":
                action = "split"
                status = "split"
            else:  # D
                action = "reject"
                status = "binned"

            if name:
                print(f"  {node['id']}: LLM→{grade}「{name}」({action})")
            else:
                print(f"  {node['id']}: LLM→{grade} ({action})")

            state.update_node(node["id"], {
                "llm_grade": grade,
                "llm_name": name,
                "llm_action": llm_action,
                "final_grade": grade,
                "action": action,
                "status": status,
            })

    rule_count = len(pending) - llm_count
    print(f"  规则确定: {rule_count}, LLM: {llm_count}")


def stage_split(state: CatalogState):
    """SPLIT: 对 C 级节点做子聚类（拆分引擎）"""
    print(f"\n[SPLIT] 拆分 C 级节点 (round={state.current_round()})...")
    split_nodes = [n for n in state.state["nodes"] if n.get("action") == "split"]
    if not split_nodes:
        print("  没有需要拆分的节点")
        return

    g = _load_globals()
    full_headlines = g['headlines']
    full_embeddings = g['embeddings']
    topic_ids = g['topic_ids']

    next_round = state.current_round() + 1
    child_counter = 0

    for node in split_nodes:
        # depth 检查：超过深度限制 → 归入残留桶
        if node.get("depth", 0) >= 2:
            state.update_node(node["id"], {
                "action": "bin_residual",
                "status": "binned",
                "final_grade": "D-fragment",
            })
            print(f"  {node['id']}: depth >= 2 → 归入残留桶")
            continue

        dl = node.get("llm_name", "")
        print(f"  {node['id']}「{dl}」: size={node['size']}, 正在拆分...")

        # 解析索引（与 _get_samples 共享逻辑）
        indices = _resolve_indices(node, topic_ids)
        if indices is None or len(indices) == 0:
            print(f"    ✗ 无法解析索引，跳过")
            continue

        # 提取子集 embedding 和标题
        sub_emb = full_embeddings[indices]
        sub_headlines = [full_headlines[i] for i in indices]

        # 自动调整 resolution
        res = 1.2 if node["size"] > 2000 else (1.0 if node["size"] > 500 else 0.8)

        try:
            sub_results = split_community(
                sub_emb, sub_headlines, resolution=res
            )
        except Exception as e:
            print(f"    ✗ 拆分失败: {e}")
            state.update_node(node["id"], {
                "action": "accept",
                "status": "accepted",
                "final_grade": "B",
            })
            continue

        if len(sub_results) <= 1:
            # 拆不出子社区 → 直接 accept
            print(f"    仅 {len(sub_results)} 个子社区，保留为 B/accept")
            state.update_node(node["id"], {
                "action": "accept",
                "status": "accepted",
                "final_grade": "B",
            })
            continue

        # 注册子社区
        child_ids = []
        for i, sr in enumerate(sub_results):
            child_id = f"v{next_round}_c{child_counter}"
            child_counter += 1
            # 将相对索引映射回全局索引
            full_indices = indices[sr["indices"]].tolist()
            child_node = {
                "id": child_id,
                "parent": node["id"],
                "depth": node.get("depth", 0) + 1,
                "size": sr["size"],
                "keywords": sr["keywords"],
                "mean_pairwise_dist": None,
                "rule_grade": None,
                "rule_signals": {},
                "llm_grade": None,
                "llm_name": None,
                "llm_action": None,
                "final_grade": None,
                "action": None,
                "children": [],
                "status": "pending",
                "_sub_indices": full_indices,
            }
            state.add_node(child_node)
            child_ids.append(child_id)

        # 更新父节点
        state.update_node(node["id"], {
            "children": child_ids,
            "status": "split",
            "action": "split_done",  # 防止下一轮重复拆分
        })
        print(f"    → {len(child_ids)} 个子社区 (v{next_round})")


def stage_check_stop(state: CatalogState) -> bool:
    """STOP_CHECK: 检查停止条件"""
    print(f"\n[STOP_CHECK] 检查停止条件...")
    passed = state.check_stopping()
    s = state.state["stopping"]
    for k, v in s.items():
        print(f"  {k}: {v}")
    print(f"  {'✅ 全部通过' if passed else '❌ 未通过'}")
    state.save()
    return passed


def stage_export(state: CatalogState):
    """EXPORT: 输出最终 catalog"""
    print(f"\n[EXPORT] 交付...")

    accepted = [n for n in state.state["nodes"]
                if n["status"] == "accepted"
                and len(n.get("children", [])) == 0]  # 只算叶子节点
    binned = state.get_binned()

    # catalog.csv
    rows = []
    for i, node in enumerate(accepted):
        rows.append({
            "topic_id": i + 1,
            "topic_name": node.get("llm_name") or "",
            "grade": node.get("final_grade", ""),
            "size": node["size"],
            "depth": node.get("depth", 0),
            "keywords": node.get("keywords", ""),
            "action": node.get("action", ""),
        })

    df_catalog = pd.DataFrame(rows)
    catalog_path = OUTPUT_DIR / "catalog.csv"
    df_catalog.to_csv(catalog_path, index=False)
    print(f"  catalog.csv: {len(df_catalog)} 条话题 → {catalog_path}")

    # 残留桶
    meta = [n for n in binned if n.get("action") == "bin_meta"]
    fragment = [n for n in binned if n.get("action") in ("bin_fragment", "bin_residual")]

    if meta:
        df_meta = pd.DataFrame([
            {"source_id": n["id"], "size": n["size"], "keywords": n.get("keywords", "")}
            for n in meta
        ])
        df_meta.to_csv(OUTPUT_DIR / "meta_bucket.csv", index=False)
        print(f"  meta_bucket.csv: {len(df_meta)} 条栏目/节目")

    if fragment:
        df_frag = pd.DataFrame([
            {"source_id": n["id"], "size": n["size"], "keywords": n.get("keywords", "")}
            for n in fragment
        ])
        df_frag.to_csv(OUTPUT_DIR / "residual_bucket.csv", index=False)
        print(f"  residual_bucket.csv: {len(df_frag)} 条残留")

    # 摘要
    total = state.state["total_headlines"]
    accepted_size = sum(n["size"] for n in accepted) if accepted else 0
    binned_size = sum(n["size"] for n in binned) if binned else 0
    residual_size = total - accepted_size - binned_size

    summary = {
        "total_headlines": total,
        "accepted_topics": len(accepted),
        "accepted_headlines": int(accepted_size),
        "accepted_pct": round(accepted_size / total * 100, 1),
        "binned_headlines": int(binned_size),
        "binned_pct": round(binned_size / total * 100, 1),
        "unaccounted": int(residual_size),
        "meta_items": len(meta),
        "residual_items": len(fragment),
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 树结构导出
    _export_tree_json(state, OUTPUT_DIR)

    print(f"\n  交付摘要:")
    for k, v in summary.items():
        print(f"    {k}: {v}")
    print(f"\n  ✅ 编目完成!")


def _export_tree_json(state: CatalogState, output_dir: Path):
    """导出层级树结构 JSON（保留 parent-child 关系）"""
    nodes = state.state["nodes"]
    node_map = {n["id"]: n for n in nodes}
    accepted_ids = {n["id"] for n in nodes if n["status"] == "accepted"}
    split_ids = {n["id"] for n in nodes if n["status"] == "split"}

    def _build_subtree(nid: str) -> dict | None:
        n = node_map.get(nid)
        if n is None or n["status"] not in ("accepted", "split"):
            return None
        name = n.get("llm_name", "") or n.get("keywords", "")[:30]
        entry = {
            "id": n["id"],
            "name": name,
            "size": n["size"],
            "depth": n["depth"],
            "status": n["status"],
        }
        children_ids = n.get("children", [])
        # 只包含被接受或处于 split 状态的子节点
        valid = [cid for cid in children_ids if cid in accepted_ids or cid in split_ids]
        children = [_build_subtree(cid) for cid in valid]
        entry["children"] = [c for c in children if c is not None]
        return entry

    # 从 parent=None 的节点开始（包括 accepted 和 split）
    roots = [n for n in nodes if n.get("parent") is None
             and n["status"] in ("accepted", "split")]
    tree = [_build_subtree(r["id"]) for r in roots if _build_subtree(r["id"]) is not None]

    # 统计叶子节点（accepted 且无 children）
    leaf_count = sum(1 for n in nodes if n["status"] == "accepted" and len(n.get("children", [])) == 0)
    total_accepted = len([n for n in nodes if n["status"] == "accepted"])

    tree_path = output_dir / "tree.json"
    with open(tree_path, "w") as f:
        json.dump({
            "tree": tree,
            "meta": {
                "root_groups": len(tree),
                "leaf_topics": leaf_count,
                "total_accepted": total_accepted,
            }
        }, f, indent=2, ensure_ascii=False)
    print(f"  tree.json: {len(tree)} 个根组, {leaf_count} 个叶话题 → {tree_path}")


# ==================== 状态机 ====================


STAGES = ["INIT", "CLUSTER", "ASSESS", "SPLIT", "CHECK_STOP", "EXPORT"]

# 状态机迭代循环
ROUND_STAGES = ["ASSESS", "SPLIT", "CHECK_STOP"]


def run(from_stage: str | None = None):
    state = CatalogState()

    if from_stage and from_stage == "INIT":
        pass
    else:
        saved = state.load()
        if saved and "nodes" in saved:
            cur_round = saved.get("round", 0)
            print(f"  📋 从 state 恢复 (round={cur_round}, {len(saved['nodes'])} 个节点)")
        else:
            from_stage = "INIT"

    start = STAGES.index(from_stage) if from_stage else 0

    # 第一阶段：INIT → CLUSTER（仅一次）
    for stage in STAGES[:3]:  # INIT, CLUSTER
        if STAGES.index(stage) < start:
            continue
        print(f"\n{'='*60}")
        print(f"  阶段: {stage}")
        print(f"{'='*60}")
        if stage == "INIT":
            stage_init(state)
        elif stage == "CLUSTER":
            stage_cluster(state)
        state.save()

    # 第二阶段：迭代轮（ASSESS → SPLIT → CHECK_STOP 循环）
    while state.is_iterating():
        cur_round = state.current_round()
        print(f"\n{'='*60}")
        print(f"  === Round {cur_round} ===")
        print(f"{'='*60}")

        # ASSESS
        stage_assess(state)
        state.save()

        # SPLIT
        stage_split(state)
        state.save()

        # CHECK_STOP
        passed = stage_check_stop(state)
        state.save()

        if passed:
            break

        state.increment_round()
        print(f"  进入 round {state.current_round()}")

    # 第三阶段：EXPORT
    print(f"\n{'='*60}")
    print(f"  阶段: EXPORT")
    print(f"{'='*60}")
    stage_export(state)
    state.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="新闻标题编目控制器")
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="从指定阶段开始（默认从头）")
    parser.add_argument("--resume", action="store_true",
                        help="从保存的 state 恢复")
    args = parser.parse_args()

    if args.resume:
        run(from_stage=None)  # 尝试恢复
    else:
        run(from_stage=args.stage)
