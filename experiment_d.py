"""
实验 D: 层次聚类 (Agglomerative + SPECTRAL) + 人工语义标注
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd, re
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 D: 层次/谱聚类")
print(f"  标题数: {len(headlines):,}")

# TF-IDF (降维到 300 维，加速层次聚类)
from sklearn.decomposition import TruncatedSVD
tfidf = TfidfVectorizer(max_features=3000, min_df=3, max_df=0.85, ngram_range=(1,2), sublinear_tf=True)
X = tfidf.fit_transform(headlines)
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)
print(f"  SVD 降维: {X.shape} → {X_reduced.shape} (解释方差: {svd.explained_variance_ratio_.sum():.2%})")

# ============ 层次聚类 (不同连接方式) ============
methods = ['ward', 'average', 'complete']
results = {}
for method in methods:
    print(f"  Agglomerative ({method})...")
    for k in [30, 50, 80]:
        ac = AgglomerativeClustering(n_clusters=k, linkage=method, metric='euclidean')
        labels = ac.fit_predict(X_reduced)
        sil = silhouette_score(X_reduced, labels, sample_size=3000)
        results[(method, k)] = (labels, sil)
        print(f"    K={k:3d}  silhouette={sil:.4f}")

# 找最优
best = max(results, key=lambda x: results[x][1])
best_method, best_k = best
best_labels, best_sil = results[best]
print(f"\n  最优层次聚类: {best_method} + K={best_k}, sil={best_sil:.4f}")

df['hier_topic'] = best_labels

# 统计分布
dist = df['hier_topic'].value_counts()
print(f"\n  Top 10 簇大小:")
for tid in dist.head(10).index:
    print(f"    簇{tid}: {dist[tid]} 条")

# ============ 谱聚类 (小规模试探) ============
# 谱聚类 O(n²) 太慢，用采样数据试探最优 K
print("\n  谱聚类采样试探最优 K...")
sample_idx = np.random.RandomState(42).choice(len(headlines), 3000, replace=False)
X_sample = X_reduced[sample_idx]

sil_scores = []
for k in [20, 30, 40, 50]:
    sc = SpectralClustering(n_clusters=k, affinity='nearest_neighbors',
                             n_neighbors=15, random_state=42)
    sc_labels = sc.fit_predict(X_sample)
    sil = silhouette_score(X_sample, sc_labels)
    sil_scores.append({'K': k, 'sil': sil})
    print(f"    K={k:2d}  silhouette={sil:.4f}")

best_sc_k = max(sil_scores, key=lambda x: x['sil'])['K']
print(f"  谱聚类最优 K: {best_sc_k}")

# 全量数据跑最优 K
print(f"  全量数据谱聚类 K={best_sc_k}...")
sc = SpectralClustering(n_clusters=best_sc_k, affinity='nearest_neighbors',
                         n_neighbors=15, random_state=42)
sc_labels_full = sc.fit_predict(X_reduced)
df['spectral_topic'] = sc_labels_full
sc_sil = silhouette_score(X_reduced, sc_labels_full)
print(f"  全量 silhouette={sc_sil:.4f}")

# ============ 对比汇总 ============
summary = pd.DataFrame({
    '方法': ['BERTopic (默认)', 'BERTopic (激进调参)', 'TF-IDF+KMeans', '层次聚类(Ward)', '谱聚类'],
    'K值/话题数': [167, '待填', best_k, best_k, best_sc_k],
    '离群率%': [47.5, '待填', 0, 0, 0],  # KMeans/LCA/层次无离群概念
    '轮廓系数': ['待填', '待填', '待填', best_sil, sc_sil],
})
summary.to_csv(OUT_DIR / "method_comparison.csv", index=False, encoding='utf-8-sig')

# 保存
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

print(f"\n  ✅ 实验 D 完成!")
print(f"  📁 {OUT_DIR}")
