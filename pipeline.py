"""
新闻标题无监督主题聚类 Pipeline
数据集: A Million News Headlines (Kaggle)
方法: BERTopic (Sentence-BERT + UMAP + HDBSCAN + c-TF-IDF)
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ============ 配置 ============
SAMPLE_SIZE = 50_000
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = Path("/Users/yuwan/code/news-clustering/data/news_headlines.csv")

print(f"[{datetime.now().strftime('%H:%M:%S')}] Step 1: 获取数据...")


# ============ Step 1: 下载 / 加载数据 ============
if not DATA_PATH.exists():
    print("  从 Kaggle 下载数据集 (kagglehub)...")
    import kagglehub
    download_path = kagglehub.dataset_download("therohk/million-headlines")
    csv_file = list(Path(download_path).glob("*.csv"))[0]
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(csv_file, DATA_PATH)
    print(f"  数据已保存到 {DATA_PATH}")
else:
    print(f"  数据已存在: {DATA_PATH}")

# 加载 & 采样
df = pd.read_csv(DATA_PATH)
print(f"  原始数据量: {len(df):,} 条")
print(f"  列名: {list(df.columns)}")

# 检测标题列名
title_col = None
for c in df.columns:
    if 'headline' in c.lower() or 'title' in c.lower():
        title_col = c
        break
if title_col is None:
    title_col = df.columns[0]  # fallback
print(f"  使用列: '{title_col}'")

# 检测日期列名
date_col = None
for c in df.columns:
    if 'date' in c.lower() or 'publish' in c.lower():
        date_col = c
        break

# 采样
if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    print(f"  随机采样 {SAMPLE_SIZE:,} 条")
df = df.reset_index(drop=True)

# 清洗: 去空
df = df.dropna(subset=[title_col])
headlines = df[title_col].astype(str).tolist()
print(f"  清洗后有效标题数: {len(headlines):,}")

# 保存采样数据
df.to_csv(OUT_DIR / "sampled_headlines.csv", index=False)


# ============ Step 2: BERTopic 建模 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: BERTopic 建模...")

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.feature_extraction.text import CountVectorizer

# 停用词
vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=2,
    max_df=0.85,
    ngram_range=(1, 2),
    max_features=5000,
)

# 主题表示: 用 KeyBERT 优化关键词
representation_model = KeyBERTInspired()

# BERTopic 模型
topic_model = BERTopic(
    language="english",
    embedding_model="all-MiniLM-L6-v2",  # 384维，轻量高效
    umap_model=None,  # 用默认参数
    hdbscan_model=None,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    min_topic_size=30,       # 每个话题至少 30 条
    nr_topics="auto",        # 自动合并相似话题
    verbose=True,
)

print("  正在拟合模型（这可能需要几分钟）...")
topics, probs = topic_model.fit_transform(headlines)

# 话题信息
topic_info = topic_model.get_topic_info()
n_topics = len(topic_info) - 1  # -1 排除 outlier (-1)
print(f"\n  发现话题数: {n_topics}")
print(f"  离群点 (未归类): {topic_info[topic_info['Topic']==-1]['Count'].values[0]:,} 条")
print(f"\n  Top 10 话题:")
print(topic_info.head(10)[['Topic', 'Count', 'Name', 'Representation']].to_string())


# ============ Step 3: 可视化 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: 生成可视化...")

# 3.1 话题间距离图 (二维投影)
fig1 = topic_model.visualize_topics()
fig1.write_html(str(OUT_DIR / "01_topic_map.html"))
print("  ✓ 01_topic_map.html (话题二维投影)")

# 3.2 话题柱状图 (Top 关键词)
fig2 = topic_model.visualize_barchart(top_n_topics=15, n_words=8)
fig2.write_html(str(OUT_DIR / "02_topic_barchart.html"))
print("  ✓ 02_topic_barchart.html (话题关键词柱状图)")

# 3.3 话题热度
fig3 = topic_model.visualize_heatmap()
fig3.write_html(str(OUT_DIR / "03_topic_heatmap.html"))
print("  ✓ 03_topic_heatmap.html (话题相似度热力图)")

# 3.4 话题层次结构
fig4 = topic_model.visualize_hierarchy()
fig4.write_html(str(OUT_DIR / "04_topic_hierarchy.html"))
print("  ✓ 04_topic_hierarchy.html (话题层次树)")

# 3.5 如果有日期列，做时间轴
if date_col is not None:
    print(f"\n  检测到日期列 '{date_col}'，生成时间轴可视化...")
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        timestamps = df[date_col].tolist()

        topics_over_time = topic_model.topics_over_time(
            headlines,
            timestamps,
            nr_bins=30,
            global_tuning=True,
        )
        fig5 = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)
        fig5.write_html(str(OUT_DIR / "05_topics_over_time.html"))
        print("  ✓ 05_topics_over_time.html (话题时间演化)")
    except Exception as e:
        print(f"  ✗ 时间轴可视化失败: {e}")
else:
    print("  (无日期列，跳过时间轴可视化)")


# ============ Step 4: 导出结果 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: 导出结果...")

# 4.1 完整话题信息
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ topic_info.csv ({len(topic_info)} 个话题)")

# 4.2 每个标题的话题标签
df['topic_id'] = topics
df['topic_name'] = df['topic_id'].map(
    lambda x: topic_info[topic_info['Topic']==x]['Name'].values[0] if x in topic_info['Topic'].values else 'Outlier'
)
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')
print(f"  ✓ headlines_with_topics.csv ({len(df)} 条)")

# 4.3 文本报告
with open(OUT_DIR / "report.md", "w", encoding='utf-8') as f:
    f.write("# 新闻标题无监督主题聚类报告\n\n")
    f.write(f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"## 数据集\n\n")
    f.write(f"- 数据集: A Million News Headlines (Kaggle)\n")
    f.write(f"- 原始数据: ~120 万条 ABC 新闻标题 (2003-2019)\n")
    f.write(f"- 采样数量: {SAMPLE_SIZE:,} 条\n")
    f.write(f"- 聚类方法: BERTopic (all-MiniLM-L6-v2 + UMAP + HDBSCAN)\n\n")
    f.write(f"## 聚类结果\n\n")
    f.write(f"- **发现话题数:** {n_topics}\n")
    f.write(f"- **离群点 (未归类):** {topic_info[topic_info['Topic']==-1]['Count'].values[0]:,} 条\n\n")
    f.write(f"### Top 15 话题\n\n")
    f.write(f"| # | Topic ID | 标题数 | 关键词 |\n")
    f.write(f"|---|---|---|---|\n")
    for _, row in topic_info[topic_info['Topic'] != -1].head(15).iterrows():
        keywords = ', '.join(row['Representation'][:6])
        f.write(f"| {_} | {row['Topic']} | {row['Count']} | {keywords} |\n")

print(f"  ✓ report.md")

# ============ 总结 ============
print(f"\n{'='*60}")
print(f"  ✅ Pipeline 完成!")
print(f"  输出目录: {OUT_DIR}")
print(f"  话题数: {n_topics}")
print(f"  离群比: {topic_info[topic_info['Topic']==-1]['Count'].values[0] / len(headlines) * 100:.1f}%")
print(f"{'='*60}")

# 列出所有输出文件
for f in sorted(OUT_DIR.glob("*")):
    size = f.stat().st_size
    if size > 1024 * 1024:
        print(f"  {f.name} ({size/1024/1024:.1f} MB)")
    elif size > 1024:
        print(f"  {f.name} ({size/1024:.0f} KB)")
    else:
        print(f"  {f.name} ({size} B)")
