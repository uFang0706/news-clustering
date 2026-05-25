"""
实验 I: 图社区发现 — k-NN 图 + Leiden 算法
============================================
核心思路:
  1. Sentence-BERT 编码全部标题 → embedding 空间
  2. 构建 k-NN 稀疏图 (cosine 相似度)
  3. Leiden 社区发现 → 每个节点天然属于一个社区，无离群概念
  4. Resolution 参数扫描 → 控制聚类粒度
  5. 评估: modularity, 话题数, silhouette, 关键词可解释性

相比 HDBSCAN 的优势:
  - 全覆盖: 每条标题都有归属，不存在离群
  - 可控粒度: resolution 参数连续调节
  - 社区质量保证: modularity 是成熟的图聚类质量指标
  - 快: k-NN 图稀疏，Leiden 近线性时间
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
import json

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_i")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 加载数据 ============
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
date_col = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else None
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 I: 图社区发现 (k-NN + Leiden)")
print(f"  标题数: {len(headlines):,}")

# ============ Step 1: 编码 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 1: Sentence-BERT 编码...")
from sentence_transformers import SentenceTransformer
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model_emb.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  Embedding 维度: {embeddings.shape}")

# ============ Step 2: 构建 k-NN 图 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: 构建 k-NN 图 (k=20)...")
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

K = 20
# 归一化到单位球面，让 Euclidean 距离等价于 Cosine 距离
# 这样 sklearn 可以用 ball_tree 加速 (O(n log n) vs O(n²))
print("  归一化 embedding...")
embeddings_norm = normalize(embeddings, norm='l2')
print(f"  构建 ball_tree 索引...")
nn = NearestNeighbors(n_neighbors=K+1, metric='euclidean', algorithm='ball_tree', n_jobs=-1)
nn.fit(embeddings_norm)
print(f"  查询 k-NN...")
distances, indices = nn.kneighbors(embeddings_norm)

# 转换为 igraph 需要的边列表 (忽略 self-loop)
import igraph as ig

edges = []
weights = []
for i in range(len(headlines)):
    for j_idx in range(1, K+1):  # 跳过自己 (j_idx=0)
        j = indices[i][j_idx]
        # Euclidean distance on normalized vectors → cosine similarity
        d = distances[i][j_idx]
        sim = 1.0 - d * d / 2.0  # cos = 1 - d²/2
        if sim > 0:  # 只用正相似度
            edges.append((i, j))
            weights.append(sim)

print(f"  边数: {len(edges):,}")
print(f"  平均度: {len(edges)/len(headlines):.1f}")

# 构建 igraph 图
g = ig.Graph(n=len(headlines), edges=edges, directed=False)
g.es['weight'] = weights

# ============ Step 3: Leiden 多分辨率扫描 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: Leiden 多分辨率扫描...")
import leidenalg

resolution_candidates = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
results = []

for res in resolution_candidates:
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=res,
        n_iterations=-1,
        seed=42,
    )
    n_communities = len(partition)
    modularity = partition.quality()
    
    # 社区大小分布
    sizes = partition.sizes()
    min_size = min(sizes)
    max_size = max(sizes)
    median_size = np.median(sizes)
    
    # 小于5的社区数 (太小没意义)
    tiny = sum(1 for s in sizes if s < 5)
    
    results.append({
        'resolution': res,
        'n_communities': n_communities,
        'modularity': modularity,
        'min_size': min_size,
        'max_size': max_size,
        'median_size': median_size,
        'tiny_communities': tiny,
    })
    
    print(f"    res={res:.1f}: {n_communities:4d} 社区, "
          f"modularity={modularity:.4f}, "
          f"size范围=[{min_size}, {median_size:.0f}, {max_size}], "
          f"tiny(<5)={tiny}")

# ============ 选择最优 resolution ============
# 规则: modularity > 0.3, 社区数在 30-150 之间, tiny占比 < 10%
print(f"\n  --- 筛选最优 resolution ---")
best_res = None
best_score = -1
for r in results:
    if r['modularity'] < 0.3:
        continue
    if r['n_communities'] < 30 or r['n_communities'] > 150:
        continue
    # 综合分数: modularity * 0.6 + 社区数合理度 * 0.4
    size_score = 1.0 - abs(r['n_communities'] - 80) / 100  # 倾向80左右
    score = r['modularity'] * 0.6 + max(0, size_score) * 0.4
    print(f"    res={r['resolution']:.1f}: score={score:.4f} (mod={r['modularity']:.4f}, n={r['n_communities']})")
    if score > best_score:
        best_score = score
        best_res = r['resolution']

if best_res is None:
    # fallback: 选 modularity 最高的
    best_res = max(results, key=lambda r: r['modularity'])['resolution']
    
print(f"\n  ✅ 最优 resolution = {best_res}")

# ============ 最终聚类 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: 最终 Leiden 聚类 (res={best_res})...")
final_partition = leidenalg.find_partition(
    g,
    leidenalg.RBConfigurationVertexPartition,
    weights='weight',
    resolution_parameter=best_res,
    n_iterations=-1,
    seed=42,
)

community_labels = np.array(final_partition.membership)
n_final = len(final_partition)
final_modularity = final_partition.quality()
final_sizes = final_partition.sizes()

print(f"  最终社区数: {n_final}")
print(f"  最终 modularity: {final_modularity:.4f}")
print(f"  社区大小: min={min(final_sizes)}, max={max(final_sizes)}, median={np.median(final_sizes):.0f}")

# 合并太小的社区 (size < 5) 到"杂项"
MIN_COMMUNITY_SIZE = 5
sizes_dict = {c: s for c, s in enumerate(final_sizes)}
valid_communities = {c for c, s in sizes_dict.items() if s >= MIN_COMMUNITY_SIZE}
tiny_communities = {c for c in sizes_dict if c not in valid_communities}

if tiny_communities:
    # 把 tiny 社区重新分配到最近的大社区
    print(f"  合并 {len(tiny_communities)} 个微小社区 (size<{MIN_COMMUNITY_SIZE})...")
    # 为新标签重映射
    new_labels = community_labels.copy()
    valid_list = sorted(valid_communities)
    label_map = {old: new for new, old in enumerate(valid_list)}
    
    for c in tiny_communities:
        mask = community_labels == c
        # 把 tiny 社区的标题归入 -1 (后续可用 centroid 重分配)
        new_labels[mask] = -1
    
    for c in valid_list:
        mask = community_labels == c
        new_labels[mask] = label_map[c]
    
    community_labels = new_labels
    n_final = len(valid_communities)

# ============ 离群重分配 (centroid-based soft assignment) ============
outlier_mask = community_labels == -1
if outlier_mask.sum() > 0:
    print(f"\n  Step 5: 重分配 {outlier_mask.sum()} 条标题到最近社区...")
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算每个社区 centroid
    centroids = {}
    for c in sorted(set(community_labels)):
        if c == -1:
            continue
        mask = community_labels == c
        centroids[c] = embeddings[mask].mean(axis=0)
    
    centroid_matrix = np.array([centroids[c] for c in sorted(centroids.keys())])
    community_list = sorted(centroids.keys())
    
    # 对每个 outlier 找最近 centroid
    outlier_embs = embeddings[outlier_mask]
    sims = cosine_similarity(outlier_embs, centroid_matrix)
    best_idx = sims.argmax(axis=1)
    best_sims = sims.max(axis=1)
    
    # 只有相似度 > 0.3 的才分配
    assignable = best_sims > 0.3
    community_labels[outlier_mask] = np.where(
        assignable,
        [community_list[i] for i in best_idx],
        -1
    )
    
    final_outliers = (community_labels == -1).sum()
    print(f"    已分配: {assignable.sum():,}, 剩余离群: {final_outliers}")
else:
    final_outliers = 0

# ============ Step 6: 关键词提取 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 6: 提取社区关键词...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9,
                         stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(headlines)
feature_names = tfidf.get_feature_names_out()

# 每个社区的 top-10 关键词
community_keywords = {}
community_sizes = {}
for c in sorted(set(community_labels)):
    if c == -1:
        continue
    mask = community_labels == c
    mean_vec = tfidf_matrix[mask].mean(axis=0).A1
    top_idx = mean_vec.argsort()[::-1][:10]
    community_keywords[int(c)] = ', '.join(feature_names[i] for i in top_idx)
    community_sizes[int(c)] = int(mask.sum())

# 打印 top-20 社区
print(f"\n  Top 20 社区:")
sorted_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)
for c, size in sorted_communities[:20]:
    print(f"    C{c:3d} ({size:5d}): {community_keywords[c]}")

# ============ Step 7: 评估 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 7: 评估...")

# silhouette score (采样 20K 计算，全量太慢)
from sklearn.metrics import silhouette_score
sample_size_sil = min(20000, len(headlines))
sample_idx = np.random.choice(len(headlines), sample_size_sil, replace=False)
sample_embs = embeddings[sample_idx]
sample_labels = community_labels[sample_idx]
# 排除 outlier
valid_sil = sample_labels != -1
if valid_sil.sum() > 1 and len(set(sample_labels[valid_sil])) > 1:
    sil = silhouette_score(sample_embs[valid_sil], sample_labels[valid_sil], metric='cosine')
else:
    sil = float('nan')
print(f"  Silhouette (cosine, {valid_sil.sum():,} samples): {sil:.4f}")

# 覆盖率
coverage = (community_labels != -1).sum() / len(headlines) * 100
print(f"  覆盖率: {coverage:.1f}%")
print(f"  最终社区数: {n_final}")
print(f"  最终 modularity: {final_modularity:.4f}")

# ============ 可视化 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 8: 可视化...")

# 8.1 社区大小分布
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sizes = [community_sizes[c] for c in sorted(community_sizes.keys())]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 柱状图
axes[0].bar(range(len(sizes)), sorted(sizes, reverse=True), color='steelblue', alpha=0.8)
axes[0].set_xlabel('Community Rank', fontsize=12)
axes[0].set_ylabel('Number of Headlines', fontsize=12)
axes[0].set_title(f'Community Size Distribution (n={n_final})', fontsize=14)
axes[0].axhline(y=np.median(sizes), color='red', linestyle='--', label=f'Median: {np.median(sizes):.0f}')
axes[0].legend()

# log-log
axes[1].loglog(range(1, len(sizes)+1), sorted(sizes, reverse=True), 'o-', markersize=3, color='steelblue')
axes[1].set_xlabel('Community Rank (log)', fontsize=12)
axes[1].set_ylabel('Size (log)', fontsize=12)
axes[1].set_title('Community Size (log-log)', fontsize=14)

plt.tight_layout()
plt.savefig(OUT_DIR / "community_sizes.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ community_sizes.png")

# 8.2 Resolution 扫描图
fig, ax1 = plt.subplots(figsize=(10, 5))
res_vals = [r['resolution'] for r in results]
n_vals = [r['n_communities'] for r in results]
mod_vals = [r['modularity'] for r in results]

ax1.plot(res_vals, n_vals, 'o-', color='steelblue', linewidth=2, markersize=8, label='Communities')
ax1.set_xlabel('Resolution', fontsize=12)
ax1.set_ylabel('Number of Communities', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')

ax2 = ax1.twinx()
ax2.plot(res_vals, mod_vals, 's-', color='coral', linewidth=2, markersize=8, label='Modularity')
ax2.set_ylabel('Modularity', color='coral', fontsize=12)
ax2.tick_params(axis='y', labelcolor='coral')

# 标记最佳
ax1.axvline(x=best_res, color='green', linestyle='--', alpha=0.5, label=f'Best res={best_res}')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Resolution vs Communities & Modularity', fontsize=14)
plt.tight_layout()
plt.savefig(OUT_DIR / "resolution_scan.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ resolution_scan.png")

# 8.3 话题地图 (UMAP 2D 投影 + 社区颜色)
print("  生成 UMAP 2D 投影...")
from umap import UMAP
umap_2d = UMAP(n_neighbors=15, n_components=2, min_dist=0.1,
                metric='cosine', random_state=42)
# 采样 10K 画图
plot_n = min(10000, len(headlines))
plot_idx = np.random.choice(len(headlines), plot_n, replace=False)
plot_embs = embeddings[plot_idx]
plot_labels = community_labels[plot_idx]
xy = umap_2d.fit_transform(plot_embs)

fig, ax = plt.subplots(figsize=(12, 10))
# 只画前 30 个最大的社区
top30 = [c for c, _ in sorted_communities[:30]]
colors = plt.cm.tab20(np.linspace(0, 1, 30))
for i, c in enumerate(top30):
    mask = plot_labels == c
    if mask.sum() > 0:
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1, alpha=0.6,
                   color=colors[i % len(colors)], label=f'C{c}')
# outliers 灰色
out_mask = plot_labels == -1
if out_mask.sum() > 0:
    ax.scatter(xy[out_mask, 0], xy[out_mask, 1], s=0.5, alpha=0.2,
               color='gray', label='Outlier')

ax.set_title(f'Leiden Communities (UMAP 2D, res={best_res})', fontsize=14)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(markerscale=5, fontsize=6, ncol=3, loc='upper right')
plt.tight_layout()
plt.savefig(OUT_DIR / "community_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ community_map.png")

# ============ 保存 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 9: 保存结果...")

df['topic_id'] = community_labels
df['topic_name'] = df['topic_id'].map(
    lambda x: community_keywords.get(x, 'Outlier')
)
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

# 社区信息
topic_info = pd.DataFrame([
    {'Topic': c, 'Count': community_sizes[c], 'Keywords': community_keywords[c]}
    for c in sorted(community_sizes.keys())
])
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')

# 汇总 JSON
summary = {
    'method': 'k-NN graph + Leiden community detection',
    'k_neighbors': K,
    'resolution': best_res,
    'n_communities': n_final,
    'modularity': float(final_modularity),
    'silhouette': float(sil) if not np.isnan(sil) else None,
    'coverage_pct': coverage,
    'n_outliers': int(final_outliers),
    'total_headlines': len(headlines),
    'embedding_model': 'all-MiniLM-L6-v2',
    'best_resolution_scan': results,
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

# 文本报告
with open(OUT_DIR / "report.md", "w", encoding='utf-8') as f:
    f.write(f"# 实验 I: 图社区发现 (k-NN + Leiden)\n\n")
    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"## 方法\n\n")
    f.write(f"- 嵌入模型: all-MiniLM-L6-v2 (384维)\n")
    f.write(f"- 图构建: k-NN (k={K}), cosine 相似度\n")
    f.write(f"- 社区发现: Leiden 算法 (RBConfiguration)\n")
    f.write(f"- 最优 resolution: {best_res}\n")
    f.write(f"- 离群重分配: centroid cosine similarity > 0.3\n\n")
    f.write(f"## 结果\n\n")
    f.write(f"- **社区数:** {n_final}\n")
    f.write(f"- **Modularity:** {final_modularity:.4f}\n")
    f.write(f"- **Silhouette (cosine):** {sil:.4f}\n")
    f.write(f"- **覆盖率:** {coverage:.1f}%\n")
    f.write(f"- **离群:** {final_outliers} 条\n\n")
    f.write(f"## Resolution 扫描\n\n")
    f.write(f"| Resolution | Communities | Modularity | Min | Max | Median | Tiny(<5) |\n")
    f.write(f"|---|---|---|---|---|---|---|\n")
    for r in results:
        f.write(f"| {r['resolution']:.1f} | {r['n_communities']} | {r['modularity']:.4f} | "
                f"{r['min_size']} | {r['max_size']} | {r['median_size']:.0f} | {r['tiny_communities']} |\n")
    f.write(f"\n## Top 20 社区\n\n")
    f.write(f"| # | Community | Size | Keywords |\n")
    f.write(f"|---|---|---|---|\n")
    for rank, (c, size) in enumerate(sorted_communities[:20], 1):
        f.write(f"| {rank} | C{c} | {size} | {community_keywords[c]} |\n")

print(f"\n  ✅ 实验 I 完成!")
print(f"  📊 社区数: {n_final}  |  modularity: {final_modularity:.4f}  |  silhouette: {sil:.4f}")
print(f"  📈 覆盖率: {coverage:.1f}%  |  离群: {final_outliers}")
print(f"  📁 {OUT_DIR}")


# ============ 对比汇总 ============
print(f"\n{'='*70}")
print(f"  实验对比汇总")
print(f"{'='*70}")
print(f"  {'实验':<6} {'方法':<28} {'话题数':<8} {'离群率':<10} {'备注'}")
print(f"  {'-'*70}")
print(f"  {'A':<6} {'BERTopic 默认':<28} {'167':<8} {'47.5%':<10} baseline")
print(f"  {'B':<6} {'BERTopic 激进调参':<28} {'84':<8} {'34.2%':<10} 精选方案")
print(f"  {'I':<6} {'k-NN + Leiden 图社区':<28} "
      f"{f'{n_final}':<8} {f'{100-coverage:.1f}%' if coverage < 100 else '0%':<10} "
      f"mod={final_modularity:.3f} sil={sil:.3f}")
print(f"{'='*70}")
