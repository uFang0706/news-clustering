"""
实验 I.2: Bayesian GMM 软聚类 — 狄利克雷过程自动选 K
======================================================
核心思路:
  1. UMAP 降维 embedding (384 → 50 维，保留局部结构)
  2. BayesianGaussianMixture (Dirichlet Process) — 自动确定聚类数
  3. 扫描 weight_concentration_prior 控制粒度
  4. 每个标题得到软分配概率 → argmax 硬分配
  5. 用熵衡量分配置信度 → 高熵 = 模糊点（类似 BERTopic outlier）

相比 Leiden 的优势:
  - 软分配: 每个点有概率分布，不只硬塞进一个簇
  - 不确定性量化: 高熵点 = 语义模糊标题，天然标记
  - 狄利克雷过程: 自动学习组件数，不需要预设 K
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
import json

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_i2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 加载数据 + 编码 (复用实验 I 的 embedding) ============
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 I.2: Bayesian GMM 软聚类")
print(f"  标题数: {len(headlines):,}")

from sentence_transformers import SentenceTransformer
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_emb.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  Embedding 维度: {embeddings.shape}")

# ============ Step 1: PCA 降维 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 1: PCA 降维 (384 → 50)...")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
pca = PCA(n_components=50, random_state=42)
embeddings_50d = pca.fit_transform(embeddings_scaled)
print(f"  降维后: {embeddings_50d.shape}")
print(f"  解释方差: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ============ Step 2: Bayesian GMM 多先验扫描 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: Bayesian GMM 扫描 weight_concentration_prior...")
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

# 只用 4 个 prior，diagonal covariance 大幅加速
priors = [0.01, 0.1, 1.0, 10.0]

results = []
for prior in priors:
    bgmm = BayesianGaussianMixture(
        n_components=200,  # 上限，DP 会自动修剪
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=prior,
        covariance_type='diag',
        max_iter=200, n_init=1,
        random_state=42, verbose=0,
        verbose_interval=10,
    )
    bgmm.fit(embeddings_50d)
    
    # 有效组件数 (权重大于阈值)
    weights = bgmm.weights_
    active = (weights > 1e-4).sum()
    
    # 硬分配
    labels = bgmm.predict(embeddings_50d)
    n_unique = len(set(labels))
    
    # 软分配熵 (衡量不确定性)
    probs = bgmm.predict_proba(embeddings_50d)
    entropies = entropy(probs.T + 1e-10)  # 加小量避免 log(0)
    mean_entropy = entropies.mean()
    
    # 高置信度点占比 (max prob > 0.5)
    max_probs = probs.max(axis=1)
    high_conf_pct = (max_probs > 0.5).sum() / len(headlines) * 100
    
    # 有效簇的 silhouette (采样) —— BUG FIX: 在 PCA 空间计算，保持与 labels 一致
    sample_n = min(15000, len(headlines))
    sample_idx = np.random.choice(len(headlines), sample_n, replace=False)
    sample_labels = labels[sample_idx]
    sample_embs = embeddings_50d[sample_idx]  # ← FIX: 用 PCA 降维后的空间
    if len(set(sample_labels)) > 1:
        sil = silhouette_score(sample_embs, sample_labels, metric='euclidean')
    else:
        sil = float('nan')
    
    # 簇大小分布
    from collections import Counter
    size_counter = Counter(labels)
    sizes = list(size_counter.values())
    
    results.append({
        'prior': prior,
        'active_components': int(active),
        'n_clusters': n_unique,
        'silhouette': float(sil),
        'mean_entropy': float(mean_entropy),
        'high_conf_pct': float(high_conf_pct),
        'min_size': min(sizes),
        'max_size': max(sizes),
        'median_size': np.median(sizes),
    })
    
    print(f"    prior={prior:5.3f}: act={int(active):3d}, n={n_unique:3d}, "
          f"sil={sil:.4f}, entropy={mean_entropy:.3f}, "
          f"high_conf={high_conf_pct:.1f}%, "
          f"size=[{min(sizes)}, {np.median(sizes):.0f}, {max(sizes)}]")

# ============ 选择最优 prior ============
print(f"\n  --- 筛选最优 ---")
# 目标: silhouette > 0, clusters 30-120, high_conf > 50%
best_prior = None
best_score = -1
for r in results:
    if np.isnan(r['silhouette']) or r['silhouette'] < -0.05:
        continue
    if r['n_clusters'] < 20 or r['n_clusters'] > 150:
        continue
    # 综合: silhouette * 0.3 + high_conf * 0.3 + 合理簇数 * 0.2 + 低熵 * 0.2
    size_score = 1.0 - abs(r['n_clusters'] - 60) / 80
    entropy_score = 1.0 - r['mean_entropy'] / 5.0  # 归一化
    score = (r['silhouette'] * 0.3 + r['high_conf_pct']/100 * 0.3 + 
             max(0, size_score) * 0.2 + max(0, entropy_score) * 0.2)
    print(f"    prior={r['prior']:.3f}: score={score:.4f} (sil={r['silhouette']:.4f}, "
          f"conf={r['high_conf_pct']:.1f}%, n={r['n_clusters']})")
    if score > best_score:
        best_score = score
        best_prior = r['prior']

if best_prior is None:
    best_prior = max(results, key=lambda r: r['silhouette'] if not np.isnan(r['silhouette']) else -1)['prior']

print(f"\n  ✅ 最优 prior = {best_prior}")

# ============ 最终 GMM ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: 最终 Bayesian GMM (prior={best_prior})...")
bgmm_final = BayesianGaussianMixture(
    n_components=200,
    weight_concentration_prior_type='dirichlet_process',
    weight_concentration_prior=best_prior,
    covariance_type='diag',
    max_iter=500, n_init=1,
    random_state=42,
)
bgmm_final.fit(embeddings_50d)

labels = bgmm_final.predict(embeddings_50d)
probs = bgmm_final.predict_proba(embeddings_50d)
max_probs = probs.max(axis=1)
entropies = entropy(probs.T + 1e-10)

n_final = len(set(labels))
print(f"  最终簇数: {n_final}")
print(f"  平均最大概率: {max_probs.mean():.3f}")
print(f"  高置信度 (>0.5): {(max_probs>0.5).sum()/len(labels)*100:.1f}%")
print(f"  高置信度 (>0.7): {(max_probs>0.7).sum()/len(labels)*100:.1f}%")
print(f"  平均熵: {entropies.mean():.3f}")

# ============ 标记低置信度点 ("软离群") ============
# 用两种阈值: max_prob < 0.3 → 离群; 0.3-0.5 → 模糊
low_conf_mask = max_probs < 0.3
fuzzy_mask = (max_probs >= 0.3) & (max_probs < 0.5)
high_conf_mask = max_probs >= 0.5

print(f"  低置信度 (<0.3): {low_conf_mask.sum():,} ({low_conf_mask.sum()/len(labels)*100:.1f}%) ← 软离群")
print(f"  模糊 (0.3-0.5): {fuzzy_mask.sum():,} ({fuzzy_mask.sum()/len(labels)*100:.1f}%)")
print(f"  高置信度 (>=0.5): {high_conf_mask.sum():,} ({high_conf_mask.sum()/len(labels)*100:.1f}%)")

# 把低置信度点归为 -1
final_labels = labels.copy()
final_labels[low_conf_mask] = -1

n_outliers = (final_labels == -1).sum()

# ============ 关键词提取 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: 提取簇关键词...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9,
                         stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(headlines)
feature_names = tfidf.get_feature_names_out()

cluster_keywords = {}
cluster_sizes = {}
for c in sorted(set(final_labels)):
    if c == -1:
        continue
    mask = final_labels == c
    mean_vec = tfidf_matrix[mask].mean(axis=0).A1
    top_idx = mean_vec.argsort()[::-1][:10]
    cluster_keywords[int(c)] = ', '.join(feature_names[i] for i in top_idx)
    cluster_sizes[int(c)] = int(mask.sum())

sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Top 20 簇:")
for c, size in sorted_clusters[:20]:
    avg_prob = max_probs[final_labels == c].mean()
    print(f"    C{c:3d} ({size:5d}, 平均置信度={avg_prob:.3f}): {cluster_keywords[c]}")

# ============ 评估 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: 评估...")
sample_n = min(15000, (final_labels != -1).sum())
sample_idx = np.random.choice(np.where(final_labels != -1)[0], sample_n, replace=False)
sample_embs = embeddings_50d[sample_idx]  # ← FIX: 用 PCA 降维后的空间
sample_lbls = final_labels[sample_idx]
if len(set(sample_lbls)) > 1:
    sil_final = silhouette_score(sample_embs, sample_lbls, metric='euclidean')
else:
    sil_final = float('nan')

coverage = (final_labels != -1).sum() / len(headlines) * 100
print(f"  Silhouette: {sil_final:.4f}")
print(f"  覆盖率 (高置信+模糊): {coverage:.1f}%")
print(f"  软离群 (置信<0.3): {n_outliers:,} ({100-coverage:.1f}%)")
print(f"  高置信 (>0.5): {high_conf_mask.sum()/len(labels)*100:.1f}%")

# ============ 可视化 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: 可视化...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 6.1 prior 扫描
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
prior_vals = [r['prior'] for r in results]

axes[0].plot(prior_vals, [r['n_clusters'] for r in results], 'o-', color='steelblue')
axes[0].set_xlabel('weight_concentration_prior')
axes[0].set_ylabel('Active Clusters')
axes[0].set_xscale('log')
axes[0].set_title('Clusters vs Prior')

axes[1].plot(prior_vals, [r['silhouette'] for r in results], 'o-', color='coral')
axes[1].axhline(y=0, color='gray', linestyle='--')
axes[1].set_xlabel('weight_concentration_prior')
axes[1].set_ylabel('Silhouette')
axes[1].set_xscale('log')
axes[1].set_title('Silhouette vs Prior')

axes[2].plot(prior_vals, [r['high_conf_pct'] for r in results], 'o-', color='green', label='High Conf %')
axes[2].plot(prior_vals, [r['mean_entropy'] for r in results], 's-', color='red', label='Mean Entropy')
axes[2].set_xlabel('weight_concentration_prior')
axes[2].set_xscale('log')
axes[2].set_title('Confidence & Entropy')
axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "prior_scan.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ prior_scan.png")

# 6.2 簇大小分布
fig, ax = plt.subplots(figsize=(10, 5))
sizes_plot = sorted([cluster_sizes[c] for c in cluster_sizes], reverse=True)
colors_plot = ['steelblue' if s > 100 else 'lightcoral' for s in sizes_plot]
ax.bar(range(len(sizes_plot)), sizes_plot, color=colors_plot, alpha=0.8)
ax.set_xlabel('Cluster Rank')
ax.set_ylabel('Size')
ax.set_title(f'GMM Cluster Size Distribution (n={n_final} active, {n_outliers} outliers)')
ax.axhline(y=np.median(sizes_plot), color='red', linestyle='--', label=f'Median: {np.median(sizes_plot):.0f}')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_sizes.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ cluster_sizes.png")

# 6.3 置信度分布
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(max_probs, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
axes[0].axvline(x=0.3, color='red', linestyle='--', label='outlier threshold')
axes[0].axvline(x=0.5, color='orange', linestyle='--', label='fuzzy threshold')
axes[0].set_xlabel('Max Probability')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Assignment Confidence')
axes[0].legend()

axes[1].hist(entropies, bins=50, color='coral', alpha=0.7, edgecolor='white')
axes[1].set_xlabel('Entropy')
axes[1].set_ylabel('Count')
axes[1].set_title('Distribution of Assignment Entropy')

plt.tight_layout()
plt.savefig(OUT_DIR / "confidence_dist.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ confidence_dist.png")

# 6.4 UMAP 2D
print("  生成 UMAP 2D 投影...")
from umap import UMAP
umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1, metric='cosine', random_state=42)
plot_n = min(10000, len(headlines))
plot_idx = np.random.choice(len(headlines), plot_n, replace=False)
xy = umap_2d.fit_transform(embeddings[plot_idx])
plot_labels = final_labels[plot_idx]

fig, ax = plt.subplots(figsize=(12, 10))
top15 = [c for c, _ in sorted_clusters[:15]]
colors = plt.cm.tab20(np.linspace(0, 1, len(top15)))
for i, c in enumerate(top15):
    mask = plot_labels == c
    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.5, alpha=0.6,
                   color=colors[i % len(colors)], label=f'C{c}')
out_mask = plot_labels == -1
if out_mask.sum() > 0:
    ax.scatter(xy[out_mask, 0], xy[out_mask, 1], s=0.5, alpha=0.2,
               color='gray', label=f'Soft Outlier ({out_mask.sum()})')
ax.set_title(f'Bayesian GMM (prior={best_prior}, {n_final} clusters)', fontsize=14)
ax.legend(markerscale=6, fontsize=7, ncol=3, loc='upper right')
plt.tight_layout()
plt.savefig(OUT_DIR / "cluster_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ cluster_map.png")

# ============ 保存 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: 保存...")

df['topic_id'] = final_labels
df['topic_name'] = df['topic_id'].map(
    lambda x: cluster_keywords.get(x, 'Soft Outlier')
)
df['confidence'] = max_probs
df['entropy'] = entropies
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

topic_info = pd.DataFrame([
    {'Topic': c, 'Count': cluster_sizes[c], 'Keywords': cluster_keywords[c],
     'Avg_Confidence': max_probs[final_labels == c].mean()}
    for c in sorted(cluster_sizes.keys())
])
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')

summary = {
    'method': 'Bayesian GMM (Dirichlet Process)',
    'embedding_model': 'all-MiniLM-L6-v2',
    'umap_components': 50,
    'best_prior': best_prior,
    'n_clusters': n_final,
    'n_soft_outliers': int(n_outliers),
    'outlier_pct': float(100 - coverage),
    'high_conf_pct': float(high_conf_mask.sum() / len(labels) * 100),
    'silhouette': float(sil_final) if not np.isnan(sil_final) else None,
    'mean_max_prob': float(max_probs.mean()),
    'mean_entropy': float(entropies.mean()),
    'prior_scan': results,
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 对比汇总
print(f"\n{'='*70}")
print(f"  实验对比汇总")
print(f"{'='*70}")
print(f"  {'实验':<6} {'方法':<30} {'话题数':<8} {'离群率':<10} {'备注'}")
print(f"  {'-'*70}")
print(f"  {'A':<6} {'BERTopic 默认':<30} {'167':<8} {'47.5%':<10} baseline")
print(f"  {'B':<6} {'BERTopic 激进调参':<30} {'84':<8} {'34.2%':<10} 精选方案")
print(f"  {'I':<6} {'k-NN + Leiden 图社区':<30} {'43':<8} {'0%':<10} sil=-0.000")
print(f"  {'I.2':<6} {'Bayesian GMM 软聚类':<30} "
      f"{f'{n_final}':<8} {f'{100-coverage:.1f}%':<10} "
      f"sil={sil_final:.3f} conf={high_conf_mask.sum()/len(labels)*100:.1f}%")
print(f"{'='*70}")

print(f"\n  ✅ 实验 I.2 完成!")
print(f"  📁 {OUT_DIR}")
