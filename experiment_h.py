"""
实验 H: 多粒度层次聚类
思路:
  - L1: 大粒度聚类 K=12 (政治/社会/体育/国际等顶层类)
  - L2: 每个 L1 类内部做细粒度子聚类
  - 离群仅在 L1 层发生，L2 层每个父节点内独立聚类
  - 话题体系天然有树形结构，方便论文画 Hierarchical 图
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_h")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
date_col = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else None
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 H: 多粒度层次聚类")
print(f"  标题数: {len(headlines):,}")

# ============ L1: 大粒度聚类 (K=12) ============
print("\n  Step 1: L1 大粒度聚类 (K=12)...")
from sklearn.cluster import KMeans

from sentence_transformers import SentenceTransformer
print("  编码标题...")
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_emb.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  编码完成: {embeddings.shape}")

# L1 用 K-Means (因为要固定 K=12)
km_l1 = KMeans(n_clusters=12, random_state=42, n_init=10)
l1_labels = km_l1.fit_predict(embeddings)
df['l1_topic'] = l1_labels

print("  L1 话题分布:")
for tid in sorted(df['l1_topic'].unique()):
    size = (df['l1_topic']==tid).sum()
    print(f"    L1_{tid}: {size:,} 条")

# ============ L2: 内部细粒度聚类 ============
print("\n  Step 2: L2 细粒度聚类 (每个 L1 类内独立 HDBSCAN)...")
l2_labels = np.full(len(headlines), -1, dtype=int)

from hdbscan import HDBSCAN as HDBSCAN_local
from umap import UMAP as UMAP_local

for l1_id in sorted(df['l1_topic'].unique()):
    mask = df['l1_topic'] == l1_id
    sub_emb = embeddings[mask]
    sub_size = mask.sum()

    # 子类数估算: 根据子样本量决定 min_cluster_size
    min_cs = max(5, sub_size // 50)  # 保证每个子类至少 5 条

    umap_sub = UMAP_local(n_neighbors=min(15, sub_size-1),
                           n_components=min(5, sub_size-1),
                           min_dist=0.0, metric='cosine', random_state=42)
    hdbscan_sub = HDBSCAN_local(min_cluster_size=min_cs, min_samples=3,
                                  prediction_data=True)

    # 降维
    if sub_size > 10:
        X_sub = umap_sub.fit_transform(sub_emb)
        sub_labels = hdbscan_sub.fit_predict(X_sub)
    else:
        sub_labels = np.zeros(sub_size, dtype=int)

    # 赋值 (全局 L2 label = l1_id * 100 + sub_labels)
    l2_labels[mask] = l1_id * 100 + sub_labels

    n_sub = len(set(sub_labels)) - (1 if -1 in sub_labels else 0)
    print(f"    L1_{l1_id} → {n_sub} 个 L2 子类 (min_cs={min_cs})")

df['l2_topic'] = l2_labels

# L2 统计
n_l2_outliers = (l2_labels % 100 == -1).sum()  # 粗略估计
l2_topic_count = len(set(l2_labels)) - (1 if -1 in l2_labels else 0)
print(f"\n  L2 总子类数: {l2_topic_count}")

# ============ 关键词提取 (每层) ============
print("\n  Step 3: 提取每层话题关键词...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=3000, min_df=2, max_df=0.9,
                         stop_words='english', ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(headlines)
feature_names = tfidf.get_feature_names_out()

def get_cluster_keywords(labels, n_clusters, tfidf_mat, feat_names, top_n=8):
    """用类内 TF-IDF 均值找关键词"""
    result = {}
    for cid in sorted(set(labels)):
        mask = labels == cid
        if mask.sum() == 0:
            continue
        # 类内 TF-IDF 均值
        mean_vec = tfidf_mat[mask].mean(axis=0).A1
        top_idx = mean_vec.argsort()[::-1][:top_n]
        result[cid] = ', '.join(feat_names[i] for i in top_idx)
    return result

# L1 关键词
l1_keywords = {}
for tid in sorted(df['l1_topic'].unique()):
    mask = df['l1_topic'] == tid
    mean_vec = tfidf_matrix[mask].mean(axis=0).A1
    top_idx = mean_vec.argsort()[::-1][:8]
    l1_keywords[tid] = ', '.join(feature_names[i] for i in top_idx)
    size = (df['l1_topic']==tid).sum()
    print(f"    L1_{tid} ({size:,}条): {l1_keywords[tid]}")

# ============ 时间轴可视化 ============
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    timestamps = df[date_col].tolist()

    # L1 时间轴
    from bertopic import BERTopic
    model_viz = BERTopic(
        language="english", embedding_model="all-MiniLM-L6-v2",
        nr_topics=None, verbose=False,
    )
    topics_viz, _ = model_viz.fit_transform(headlines)
    tot = model_viz.topics_over_time(headlines, timestamps, topics=topics_viz, nr_bins=30)
    fig = model_viz.visualize_topics_over_time(tot, top_n_topics=12)
    fig.write_html(str(OUT_DIR / "topics_over_time.html"))

# ============ 保存 ============
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

# 保存层次结构
import json
hierarchy = {}
for tid in sorted(df['l1_topic'].unique()):
    l1_mask = df['l1_topic'] == tid
    l2_ids = sorted(set(df.loc[l1_mask, 'l2_topic'].values))
    hierarchy[f"L1_{tid}"] = {
        "keywords": l1_keywords[tid],
        "count": int(l1_mask.sum()),
        "l2_children": {int(c): "" for c in l2_ids if c != tid*100-1}
    }

with open(OUT_DIR / "hierarchy.json", "w") as f:
    json.dump(hierarchy, f, indent=2, ensure_ascii=False)

# 汇总统计
l1_outlier_pct = (df['l1_topic']==-1).sum() / len(df) * 100
l2_outlier_pct = (df['l2_topic']%100 == -1).sum() / len(df) * 100
print(f"\n  ✅ 实验 H 完成!")
print(f"  L1 离群率: {l1_outlier_pct:.1f}%")
print(f"  L2 层总子类: {l2_topic_count}")
print(f"  📁 {OUT_DIR}")
