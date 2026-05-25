"""
精简版 Leiden 聚类 — 只跑候选生成，不跑多分辨率扫描和可视化

用法:
    HF_HUB_OFFLINE=1 python3 cluster_leiden_fast.py
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_i")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("[Leiden 快速版] 加载数据...")
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
headlines = df[title_col].astype(str).tolist()
print(f"  标题数: {len(headlines):,}")

# === Step 1: Embedding ===
print("\n[1/3] MiniLM 编码...")
from sentence_transformers import SentenceTransformer
t0 = time.time()
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  维度: {embeddings.shape}, 耗时: {time.time()-t0:.1f}s")
np.save(OUTPUT_DIR / "embeddings.npy", embeddings)

# === Step 2: k-NN 图 ===
print("\n[2/3] k-NN 构图 (k=20)...")
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

t0 = time.time()
embeddings_norm = normalize(embeddings, norm='l2')
nn = NearestNeighbors(n_neighbors=21, metric='euclidean', algorithm='ball_tree', n_jobs=-1)
nn.fit(embeddings_norm)
distances, indices = nn.kneighbors(embeddings_norm)
print(f"  构图完成, 耗时: {time.time()-t0:.1f}s")

import igraph as ig
edges = []
weights = []
for i in range(len(headlines)):
    for j_idx in range(1, 21):
        j = indices[i][j_idx]
        d = distances[i][j_idx]
        sim = 1.0 - d * d / 2.0
        if sim > 0:
            edges.append((i, j))
            weights.append(sim)

g = ig.Graph(n=len(headlines), edges=edges, directed=False)
g.es['weight'] = weights
print(f"  边数: {len(edges):,}, 平均度: {len(edges)/len(headlines):.1f}")

# === Step 3: Leiden (固定 resolution=1.0) ===
print("\n[3/3] Leiden (res=1.0)...")
import leidenalg
t0 = time.time()
partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    weights='weight',
    resolution_parameter=1.0,
    n_iterations=-1,
    seed=42,
)
print(f"  Leiden 完成, 耗时: {time.time()-t0:.1f}s")

sizes = partition.sizes()
n_comm = len(sizes)
max_size = max(sizes)
print(f"  社区数: {n_comm}, 最大社区: {max_size:,}, 中位: {np.median(sizes):.0f}")

labels = np.array(partition.membership)
topic_ids = pd.Series(labels, name="topic_id")
result = pd.concat([df, topic_ids], axis=1)

# 用 TF-IDF 提关键词
print("\n  提取 TF-IDF 关键词...")
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9,
                         stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(headlines)
feature_names = tfidf.get_feature_names_out()

topic_info = []
for cid in range(n_comm):
    mask = labels == cid
    size = int(mask.sum())
    mean_vec = tfidf_matrix[mask].mean(axis=0).A1
    top_idx = mean_vec.argsort()[::-1][:10]
    keywords = ', '.join(feature_names[i] for i in top_idx)
    topic_info.append({"Topic": cid, "Count": size, "Keywords": keywords})

df_info = pd.DataFrame(topic_info)
df_info.to_csv(OUTPUT_DIR / "topic_info.csv", index=False)
result.to_csv(OUTPUT_DIR / "headlines_with_topics.csv", index=False)

summary = {
    "method": "Leiden (k-NN, resolution=1.0)",
    "n_communities": n_comm,
    "max_community_size": int(max_size),
    "max_pct": round(max_size / len(headlines) * 100, 1),
    "median_size": int(np.median(sizes)),
}
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ 完成! {n_comm} 个社区")
print(f"  最大社区: {max_size:,} ({summary['max_pct']}%)")
print(f"  输出: {OUTPUT_DIR}")
