"""
split_engine.py — 对 C 级社区做子聚类拆分（Leiden 再聚类）

用法:
    from split_engine import split_community
    sub_communities = split_community(
        embeddings=embeddings_subset,
        headlines=headlines_subset,
        resolution=1.2,
    )
    # 返回 [(label, size, keywords, top_samples, boundary_samples), ...]
"""

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import igraph as ig
import leidenalg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ==================== 核心拆分函数 ====================


def split_community(
    embeddings: np.ndarray,
    headlines: list[str],
    resolution: float = 1.2,
    k: int = 15,
    seed: int = 42,
    min_community_size: int = 5,
) -> list[dict]:
    """
    对一个社区的子集做 Leiden 再聚类。

    Args:
        embeddings: (N, D) 子集 embedding 矩阵
        headlines: N 条标题列表
        resolution: Leiden resolution 参数（越大社区越多）
        k: k-NN 邻居数
        seed: 随机种子
        min_community_size: 最小社区大小（小于此并入残留桶）

    Returns:
        [{"label": int, "size": int, "keywords": str,
          "indices": np.ndarray, "top_samples": [str], "boundary_samples": [str]}, ...]
    """
    n = len(headlines)
    if n < 10:
        # 太小了，无需拆分
        return [{"label": 0, "size": n, "keywords": ", ".join(headlines[:5]),
                 "indices": np.arange(n), "top_samples": headlines[:3],
                 "boundary_samples": headlines[-2:]}]

    # 动态调整 resolution：大社区更细，小社区更粗
    auto_res = 0.6 if n < 30 else (0.8 if n < 100 else resolution)
    if resolution == 1.2:  # 未手动指定时使用自动值
        resolution = auto_res
    t0 = time.time()
    print(f"    [拆分] n={n}, k={k}, res={resolution}")

    # Step 1: k-NN 构图
    embeddings_norm = normalize(embeddings, norm="l2")
    nn_model = NearestNeighbors(
        n_neighbors=min(k + 1, n),
        metric="euclidean",
        algorithm="ball_tree",
        n_jobs=-1,
    )
    nn_model.fit(embeddings_norm)
    distances, indices = nn_model.kneighbors(embeddings_norm)

    edges = []
    weights = []
    for i in range(n):
        n_neighbors = min(k + 1, n)
        for j_idx in range(1, n_neighbors):
            j = indices[i][j_idx]
            d = distances[i][j_idx]
            sim = 1.0 - d * d / 2.0
            if sim > 0:
                edges.append((i, j))
                weights.append(sim)

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    print(f"      构图: {len(edges)} 边, 耗时 {time.time()-t0:.1f}s")

    # Step 2: Leiden
    t1 = time.time()
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        n_iterations=-1,
        seed=seed,
    )
    labels = np.array(partition.membership)
    print(f"      Leiden: {len(set(labels))} 子社区, 耗时 {time.time()-t1:.1f}s")

    # Step 3: TF-IDF 关键词
    t2 = time.time()

    # 过滤小于 min_community_size 的子社区
    unique_labels = sorted(set(labels))
    valid_labels = []
    for lbl in unique_labels:
        mask = labels == lbl
        if int(mask.sum()) >= min_community_size:
            valid_labels.append(lbl)

    if len(valid_labels) == 0:
        # 全太小了，合为一个
        return [{"label": 0, "size": n, "keywords": ", ".join(headlines[:5]),
                 "indices": np.arange(n), "top_samples": headlines[:3],
                 "boundary_samples": headlines[-2:]}]

    # 只对有效子社区提取关键词
    headlines_arr = np.array(headlines)
    tfidf = TfidfVectorizer(max_features=2000, min_df=1, max_df=0.95,
                            stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(headlines)
    feature_names = tfidf.get_feature_names_out()

    results = []
    for lbl in valid_labels:
        mask = labels == lbl
        sub_indices = np.where(mask)[0]
        sub_size = int(mask.sum())

        # 关键词
        mean_vec = tfidf_matrix[mask].mean(axis=0).A1
        top_idx = mean_vec.argsort()[::-1][:10]
        keywords = ", ".join(feature_names[i] for i in top_idx)

        # 中心排序样本
        sub_emb = embeddings[sub_indices]
        centroid = sub_emb.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-10:
            sims = sub_emb @ centroid
        else:
            sims = np.ones(sub_emb.shape[0])  # 零向量 → 等距
        order = np.argsort(-sims)

        sub_headlines = headlines_arr[sub_indices].tolist()
        top_samples = [sub_headlines[i] for i in order[:5]]
        boundary_samples = [sub_headlines[i] for i in order[-3:]]

        results.append({
            "label": lbl,
            "size": sub_size,
            "keywords": keywords,
            "indices": sub_indices,
            "top_samples": top_samples,
            "boundary_samples": boundary_samples,
        })

    results.sort(key=lambda r: -r["size"])
    for i, r in enumerate(results):
        print(f"      子社区 {i}: {r['size']}条, kw={r['keywords'][:60]}")
    print(f"      关键词提取耗时 {time.time()-t2:.1f}s")

    return results


# ==================== 命令行测试 ====================


def main():
    """从命令行测试拆分引擎"""
    import json

    # 加载 embeddings
    emb_path = Path("output_exp_i/embeddings.npy")
    embeddings = np.load(emb_path)

    # 加载 headlines
    df = pd.read_csv("output/sampled_headlines.csv")
    title_col = [c for c in df.columns if "headline" in c.lower() or "title" in c.lower()][0]
    headlines = df[title_col].astype(str).tolist()

    # 加载社区分配
    df_comm = pd.read_csv("output_exp_i/headlines_with_topics.csv")
    topic_col = [c for c in df_comm.columns if "topic" in c.lower()][0]
    topic_ids = df_comm[topic_col].values

    # 读取 catalog state 中 split 状态的节点
    from catalog_state import CatalogState
    state = CatalogState()
    state.load()
    split_nodes = [n for n in state.state["nodes"] if n.get("action") == "split"]

    print(f"待拆分社区: {len(split_nodes)}")
    for node in split_nodes:
        # 找到这个节点对应的实际 topic_id
        node_idx = int(node["id"].split("_c")[1])
        unique_ids = np.unique(topic_ids)
        unique_ids = unique_ids[unique_ids >= 0]
        sorted_ids = sorted(unique_ids)
        if node_idx >= len(sorted_ids):
            print(f"  {node['id']}: 索引越界({node_idx} >= {len(sorted_ids)})")
            continue
        cid = sorted_ids[node_idx]
        mask = topic_ids == cid
        indices = np.where(mask)[0]

        print(f"\n  {node['id']}「{node.get('llm_name','?')}」(topic_id={cid}, {len(indices)}条)")

        sub_emb = embeddings[indices]
        sub_headlines = [headlines[i] for i in indices]

        # 自动调整 resolution
        res = 1.0 if len(indices) > 1000 else 0.8
        results = split_community(sub_emb, sub_headlines, resolution=res)

        print(f"    → {len(results)} 个子社区")


if __name__ == "__main__":
    main()
