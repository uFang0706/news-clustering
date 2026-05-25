"""
实验 E: 升级 Embedding 模型 - all-mpnet-base-v2
比 all-MiniLM-L6-v2 强在: 768维, 更好的语义理解, MS MARCO 排行榜前列
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_e")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
date_col = [c for c in df.columns if 'date' in c.lower()][0] if any('date' in c.lower() for c in df.columns) else None
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 E: 升级 embedding → all-mpnet-base-v2")
print(f"  标题数: {len(headlines):,}")

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
    embedding_model="all-mpnet-base-v2",  # ← 升级点
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    min_topic_size=15,
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

print(f"\n  📁 {OUT_DIR}")
print(f"  话题数: {n_topics}  离群率: {outlier_pct:.1f}%")
