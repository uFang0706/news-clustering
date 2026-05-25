"""
实验 F: 短文本增强 - 相似标题拼接策略
思路: 把语义相近的标题拼接成段落，增加每条样本的信息密度
实现: 用已训练的 BERTopic embedding 两两相似度 > 0.7 的按时间窗口拼接
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_f")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
date_col = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else None
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 F: 短文本增强拼接")
print(f"  原始标题数: {len(headlines):,}")

# ============ 策略: TF-IDF 相似标题拼接 ============
print("  Step 1: TF-IDF 快速近邻拼接...")
tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9)
X_tfidf = tfidf.fit_transform(headlines)

# 对每条，找最相似的 1-2 个邻居（同一天，TF-IDF 相似度 > 0.4）
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=3, metric='cosine', algorithm='brute')
nn.fit(X_tfidf)
distances, indices = nn.kneighbors(X_tfidf)

# 拼接增强文本
enhanced = []
assignments = []  # 记录每条原始标题属于哪个增强块
block_id = 0

for i in range(len(headlines)):
    # 找相似邻居（排除自己，相似度 > 0.4 才拼接）
    neighbors = [(i, 1.0)]  # 自己
    for j, d in zip(indices[i][1:], distances[i][1:]):
        if d < 0.4:  # cosine distance < 0.4 → similarity > 0.6
            neighbors.append((j, 1 - d))

    # 按相似度加权拼接（取 top-3）
    neighbors = sorted(neighbors, key=lambda x: x[1], reverse=True)[:3]
    block_text = ' | '.join([headlines[idx] for idx, _ in neighbors])
    enhanced.append(block_text)
    assignments.append(block_id)
    block_id += 1

print(f"  增强后文本数: {len(enhanced):,}")
print(f"  示例: {enhanced[0]}")

# ============ BERTopic 聚类增强文本 ============
print(f"\n  Step 2: BERTopic 聚类增强文本...")
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN

vectorizer_model = CountVectorizer(
    stop_words="english", min_df=2, max_df=0.85,
    ngram_range=(1, 2), max_features=6000,
)
umap_model = UMAP(n_neighbors=10, n_components=5, min_dist=0.0,
                   metric='cosine', random_state=42)
hdbscan_model = HDBSCAN(min_cluster_size=15, min_samples=3,
                         cluster_selection_epsilon=0.0, prediction_data=True)
representation_model = KeyBERTInspired()

model = BERTopic(
    language="english",
    embedding_model="all-MiniLM-L6-v2",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    min_topic_size=15,
    nr_topics="auto",
    verbose=True,
)

topics, probs = model.fit_transform(enhanced)
topic_info = model.get_topic_info()
n_topics = len(topic_info) - 1
n_outliers = topic_info[topic_info['Topic']==-1]['Count'].values[0]
outlier_pct = n_outliers / len(enhanced) * 100

print(f"\n  ✅ 话题数: {n_topics}  | 离群: {n_outliers:,} ({outlier_pct:.1f}%)")

# 可视化
fig1 = model.visualize_topics()
fig1.write_html(str(OUT_DIR / "01_topic_map.html"))
fig2 = model.visualize_barchart(top_n_topics=20, n_words=8)
fig2.write_html(str(OUT_DIR / "02_topic_barchart.html"))

# 时间轴（用原始日期）
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    timestamps = df[date_col].tolist()
    tot = model.topics_over_time(enhanced, timestamps, nr_bins=30)
    fig3 = model.visualize_topics_over_time(tot, top_n_topics=12)
    fig3.write_html(str(OUT_DIR / "03_topics_over_time.html"))

# 导出
df['topic_id'] = topics
df['topic_name'] = df['topic_id'].map(
    lambda x: topic_info[topic_info['Topic']==x]['Name'].values[0]
    if x in topic_info['Topic'].values else 'Outlier'
)
df['enhanced_text'] = enhanced
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')

print(f"\n  📁 {OUT_DIR}")
print(f"  话题数: {n_topics}  离群率: {outlier_pct:.1f}%")
