"""
实验 B: BERTopic 调参优化 - 降低离群率
策略:
  1. 更小的 min_topic_size (20 → 10)
  2. 自定义 HDBSCAN 参数 (更激进)
  3. 降低 UMAP 维度容忍度
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

OUT_DIR  = Path("/Users/yuwan/code/news-clustering/output_exp_b")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载数据
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
date_col  = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else None
headlines = df[title_col].astype(str).tolist()

print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 B: BERTopic 激进调参")
print(f"  标题数: {len(headlines):,}")

# 停用词
vectorizer_model = CountVectorizer(
    stop_words="english", min_df=2, max_df=0.85,
    ngram_range=(1, 2), max_features=6000,
)

# UMAP: 更激进降维，保留更多局部结构
umap_model = UMAP(
    n_neighbors=10,      # 默认15 → 10，更局部
    n_components=5,      # 默认15 → 5，减少信息损失
    min_dist=0.0,       # 默认0.1 → 0，簇更紧凑
    metric='cosine',
    random_state=42,
)

# HDBSCAN: 更激进
hdbscan_model = HDBSCAN(
    min_cluster_size=10,  # 默认30 → 10，允许小簇
    min_samples=3,        # 新增，降低密度要求
    cluster_selection_epsilon=0.0,
    gen_min_span_tree=True,
    prediction_data=True,
)

representation_model = KeyBERTInspired()

model = BERTopic(
    language="english",
    embedding_model="all-MiniLM-L6-v2",
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    min_topic_size=10,
    nr_topics="auto",
    verbose=True,
)

topics, probs = model.fit_transform(headlines)
topic_info = model.get_topic_info()
n_topics = len(topic_info) - 1
n_outliers = topic_info[topic_info['Topic']==-1]['Count'].values[0]
outlier_pct = n_outliers / len(headlines) * 100

print(f"\n  ✅ 话题数: {n_topics}  | 离群: {n_outliers:,} ({outlier_pct:.1f}%)")

# 可视化
fig1 = model.visualize_topics()
fig1.write_html(str(OUT_DIR / "01_topic_map.html"))
fig2 = model.visualize_barchart(top_n_topics=20, n_words=8)
fig2.write_html(str(OUT_DIR / "02_topic_barchart.html"))
fig3 = model.visualize_hierarchy()
fig3.write_html(str(OUT_DIR / "03_topic_hierarchy.html"))

# 时间轴
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    timestamps = df[date_col].tolist()
    tot = model.topics_over_time(headlines, timestamps, nr_bins=30)
    fig4 = model.visualize_topics_over_time(tot, top_n_topics=12)
    fig4.write_html(str(OUT_DIR / "04_topics_over_time.html"))

# 导出
df['topic_id'] = topics
df['topic_name'] = df['topic_id'].map(
    lambda x: topic_info[topic_info['Topic']==x]['Name'].values[0]
    if x in topic_info['Topic'].values else 'Outlier'
)
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')

print(f"\n  📁 输出: {OUT_DIR}")
print(f"  话题数: {n_topics}  离群率: {outlier_pct:.1f}%")
