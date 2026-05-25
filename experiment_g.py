"""
实验 G: 离群标题二次归类 (Soft Re-assignment)
思路: BERTopic 的离群标题，用余弦相似度强行归入最接近的话题
流程:
  1. 取每个话题的 centroid embedding (该话题下所有标题 embedding 的均值)
  2. 对每个离群标题，计算与所有 centroid 的余弦相似度
  3. 相似度 > threshold 的强行归入最接近话题
  4. 调整 threshold 找到最优
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_g")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
headlines = df[df.columns[0]].astype(str).tolist()

# 用实验 B 的结果（84个话题，离群率34.2%）
df_b = pd.read_csv("/Users/yuwan/code/news-clustering/output_exp_b/headlines_with_topics.csv")
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 G: 离群二次归类")
print(f"  当前离群数: {(df_b['topic_id']==-1).sum():,} 条")

# ============ 重新编码全部标题 ============
print("  Step 1: Sentence-BERT 编码全部标题...")
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
all_embeddings = model_emb.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  编码完成: {all_embeddings.shape}")

# ============ 计算话题 centroid ============
print("  Step 2: 计算每个话题的 centroid...")
topic_ids = df_b['topic_id'].values
unique_topics = sorted(set(topic_ids))

centroids = {}
for tid in unique_topics:
    if tid == -1:
        continue
    mask = topic_ids == tid
    centroids[tid] = all_embeddings[mask].mean(axis=0)

centroid_matrix = np.array([centroids[t] for t in sorted(centroids.keys())])
topic_list = sorted(centroids.keys())
print(f"  话题 centroid 数: {len(centroids)}")

# ============ 批量扫描最优 threshold ============
print("  Step 3: 扫描最优相似度阈值...")
outlier_mask = topic_ids == -1
outlier_embeddings = all_embeddings[outlier_mask]

# 批量计算离群点 vs 所有 centroid 的相似度
similarities = cosine_similarity(outlier_embeddings, centroid_matrix)
# 每行最大值和对应的话题
max_sims = similarities.max(axis=1)
best_topics = np.array([topic_list[i] for i in similarities.argmax(axis=1)])

print(f"  离群标题相似度分布:")
for thresh in [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]:
    n_reassigned = (max_sims >= thresh).sum()
    pct = n_reassigned / len(max_sims) * 100
    new_outliers = len(max_sims) - n_reassigned
    new_outlier_pct = new_outliers / len(headlines) * 100
    print(f"    threshold={thresh:.2f}: 可归入 {n_reassigned:,} ({pct:.1f}%)  "
          f"→ 剩余离群 {new_outliers:,} (总离群率 {new_outlier_pct:.1f}%)")

# 用 threshold=0.70 作为默认（语义接近才归，不乱塞）
best_thresh = 0.70
reassigned_mask = max_sims >= best_thresh
n_reassigned = reassigned_mask.sum()
new_topic_ids = topic_ids.copy()
new_topic_ids[outlier_mask] = np.where(
    reassigned_mask,
    best_topics,
    -1
)
n_remaining_outliers = (new_topic_ids == -1).sum()
new_outlier_pct = n_remaining_outliers / len(headlines) * 100

print(f"\n  ✅ 最终: 归入 {n_reassigned:,} 条，剩余离群 {n_remaining_outliers:,} ({new_outlier_pct:.1f}%)")

# ============ 验证: 归入前后的话题分布变化 ============
print(f"\n  话题分布变化 (归入前后):")
df_b['new_topic_id'] = new_topic_ids
before = df_b[df_b['topic_id']!=-1]['topic_id'].nunique()
after = df_b[df_b['new_topic_id']!=-1]['new_topic_id'].nunique()
print(f"    有标题的话题数: {before} → {after}")

# ============ 保存结果 ============
df_b['final_topic_id'] = new_topic_ids
df_b.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

# 统计报告
summary = {
    '原始离群率': f"{(topic_ids==-1).sum()/len(headlines)*100:.1f}%",
    '二次归类阈值': best_thresh,
    '成功归入': f"{n_reassigned:,} 条",
    '剩余离群': f"{n_remaining_outliers:,} 条",
    '最终离群率': f"{new_outlier_pct:.1f}%",
    '离群率下降': f"{(topic_ids==-1).sum()/len(headlines)*100 - new_outlier_pct:.1f}%"
}
pd.DataFrame([summary]).to_csv(OUT_DIR / "reassignment_summary.csv", index=False)
print(f"\n  📁 {OUT_DIR}")
print(f"  ✅ 实验 G 完成!")
