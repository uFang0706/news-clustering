"""
实验 J: LLM 话题质量评估与智能合并
===================================
思路:
  1. 取最优 BERTopic 结果 (实验 B, 84 话题)
  2. 计算话题对之间的 centroid cosine 相似度
  3. 对相似度 0.7-0.95 的话题对，让 LLM 判断是否应合并
  4. LLM 生成合并后话题的可读名称
  5. 对剩余离群标题，LLM 语义分类到最近话题

LLM: MiniMax-M2.7 (via anthropic-compatible API)
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json, time, os

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_j")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============ 加载数据 ============
from sentence_transformers import SentenceTransformer

df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
headlines = df[title_col].astype(str).tolist()

print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 J: LLM 话题评估与智能合并")
print(f"  标题数: {len(headlines):,}")

# 加载实验 B 结果
df_b = pd.read_csv("/Users/yuwan/code/news-clustering/output_exp_b/headlines_with_topics.csv")
topic_info_b = pd.read_csv("/Users/yuwan/code/news-clustering/output_exp_b/topic_info.csv")

# 只取有效话题 (非 outlier)
valid_topics = topic_info_b[topic_info_b['Topic'] != -1]
print(f"  实验 B 有效话题数: {len(valid_topics)}")

# ============ Step 1: 计算话题 centroid ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 1: 编码 + 计算 centroid...")
model_emb = SentenceTransformer("all-MiniLM-L6-v2")
all_embeddings = model_emb.encode(headlines, show_progress_bar=True, batch_size=256)

topic_ids_b = df_b['topic_id'].values
centroids = {}
topic_samples = {}  # 每个话题存几条样本

for _, row in valid_topics.iterrows():
    tid = row['Topic']
    mask = topic_ids_b == tid
    centroids[tid] = all_embeddings[mask].mean(axis=0)
    # 存 5 条随机样本
    sample_indices = np.where(mask)[0]
    if len(sample_indices) > 5:
        sample_indices = np.random.choice(sample_indices, 5, replace=False)
    topic_samples[tid] = [headlines[i] for i in sample_indices]

# 建议的话题名称 (取关键词前4个)
topic_names = {}
for _, row in valid_topics.iterrows():
    kw = row['Representation']
    if isinstance(kw, str):
        # 可能存的是字符串形式的列表
        try:
            kw_list = eval(kw)[:4]
        except:
            kw_list = kw.split(', ')[:4]
    else:
        kw_list = kw[:4] if isinstance(kw, list) else []
    topic_names[row['Topic']] = ', '.join(kw_list)

print(f"  计算了 {len(centroids)} 个 centroid")

# ============ Step 2: 找相似话题对 (merge candidates) ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 2: 找候选合并对...")
from sklearn.metrics.pairwise import cosine_similarity

topic_list = sorted(centroids.keys())
centroid_matrix = np.array([centroids[t] for t in topic_list])
sim_matrix = cosine_similarity(centroid_matrix)

# 找相似度 0.65-0.95 的对 (排除自身)
candidates = []
for i in range(len(topic_list)):
    for j in range(i+1, len(topic_list)):
        sim = sim_matrix[i, j]
        if 0.65 <= sim < 0.95:
            t1, t2 = topic_list[i], topic_list[j]
            size1 = valid_topics[valid_topics['Topic']==t1]['Count'].values[0]
            size2 = valid_topics[valid_topics['Topic']==t2]['Count'].values[0]
            candidates.append((sim, t1, t2, int(size1), int(size2)))

candidates.sort(reverse=True)  # 最相似的在前
print(f"  候选合并对: {len(candidates)} 对 (相似度 0.65-0.95)")
print(f"  Top 10 候选:")
for sim, t1, t2, s1, s2 in candidates[:10]:
    print(f"    sim={sim:.3f}: T{t1}({s1}): {topic_names[t1][:60]}")
    print(f"             T{t2}({s2}): {topic_names[t2][:60]}")

# ============ Step 3: LLM 评估 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 3: LLM 评估候选合并...")

# MiniMax API
import requests

API_KEY = os.environ.get("MINIMAX_CN_API_KEY") or os.environ.get("MINIMAX_API_KEY")
if not API_KEY:
    # 尝试从 .env 读取
    env_path = Path.home() / ".hermes" / ".env"
    if env_path.exists():
        for line in env_path.read_text().split('\n'):
            if line.startswith('MINIMAX_CN_API_KEY='):
                API_KEY = line.split('=', 1)[1].strip().strip('"').strip("'")
                break

if not API_KEY:
    print("  ⚠️ 未找到 MiniMax API Key，跳过 LLM 评估")
    merge_decisions = {}
else:
    BASE_URL = "https://api.minimaxi.com/anthropic/v1/messages"
    
    # 只取 top-30 候选对做 LLM 评估
    eval_candidates = candidates[:min(30, len(candidates))]
    merge_decisions = {}
    
    for idx, (sim, t1, t2, s1, s2) in enumerate(eval_candidates):
        # 构造 prompt
        k1 = topic_names[t1]
        k2 = topic_names[t2]
        samples1 = ' | '.join(topic_samples[t1])
        samples2 = ' | '.join(topic_samples[t2])
        
        prompt = f"""You are evaluating whether two news headline topics should be merged.

Topic A (ID={t1}, {s1} headlines):
  Keywords: {k1}
  Sample headlines: {samples1}

Topic B (ID={t2}, {s2} headlines):
  Keywords: {k2}
  Sample headlines: {samples2}

Centroid cosine similarity: {sim:.3f}

Should these be merged into one topic? Reply with ONLY one word: YES or NO"""

        try:
            resp = requests.post(
                BASE_URL,
                json={
                    "model": "MiniMax-M2.7",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 10,
                    "temperature": 0.0,
                },
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=30,
            )
            answer = resp.json()['content'][0]['text'].strip().upper()
            merge_decisions[(t1, t2)] = answer.startswith('YES')
            
            if (idx + 1) % 5 == 0:
                print(f"    [{idx+1}/{len(eval_candidates)}] 评估中...")
            
        except Exception as e:
            print(f"    ✗ T{t1} vs T{t2}: LLM 调用失败 ({e})")
            merge_decisions[(t1, t2)] = False
        
        time.sleep(0.3)  # rate limit
    
    n_merge = sum(1 for v in merge_decisions.values() if v)
    print(f"  LLM 建议合并: {n_merge}/{len(eval_candidates)} 对")

# ============ Step 4: 执行合并 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 4: 执行合并...")

# Union-Find 构建合并组
parent = {t: t for t in topic_list}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(x, y):
    px, py = find(x), find(y)
    if px != py:
        parent[px] = py

for (t1, t2), should_merge in merge_decisions.items():
    if should_merge:
        union(t1, t2)

# 收集合并组
groups = defaultdict(list)
for t in topic_list:
    root = find(t)
    groups[root].append(t)

merge_groups = {k: v for k, v in groups.items() if len(v) > 1}
print(f"  合并组数: {len(merge_groups)}")
for root, members in merge_groups.items():
    total_size = sum(valid_topics[valid_topics['Topic']==m]['Count'].values[0] for m in members)
    names = [topic_names[m] for m in members]
    print(f"    → {total_size}条: {' | '.join(names[:3])}")

# 应用合并: 重新映射 topic_id
old_to_new = {}
next_id = 0
for _, members in merge_groups.items():
    for m in members:
        old_to_new[m] = next_id
    next_id += 1
for t in topic_list:
    if t not in old_to_new:
        old_to_new[t] = next_id
        next_id += 1

new_topic_ids = np.array([old_to_new.get(t, t) for t in topic_ids_b])
n_merged_topics = len(set(new_topic_ids[new_topic_ids != -1]))
print(f"  合并后话题数: {n_merged_topics} (原始: {len(valid_topics)})")

# ============ Step 5: 重新提取关键词 ============
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Step 5: 重新提取关键词...")
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9,
                         stop_words='english', ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(headlines)
feature_names = tfidf.get_feature_names_out()

new_keywords = {}
new_sizes = {}
for c in sorted(set(new_topic_ids)):
    if c == -1:
        continue
    mask = new_topic_ids == c
    mean_vec = tfidf_matrix[mask].mean(axis=0).A1
    top_idx = mean_vec.argsort()[::-1][:8]
    new_keywords[int(c)] = ', '.join(feature_names[i] for i in top_idx)
    new_sizes[int(c)] = int(mask.sum())

sorted_new = sorted(new_sizes.items(), key=lambda x: x[1], reverse=True)
print(f"\n  Top 15 合并后话题:")
for c, size in sorted_new[:15]:
    print(f"    C{c:3d} ({size:5d}): {new_keywords[c]}")

# ============ Step 6: 离群统计 ============
new_outliers = (new_topic_ids == -1).sum()
new_outlier_pct = new_outliers / len(headlines) * 100
print(f"\n  离群: {new_outliers:,} ({new_outlier_pct:.1f}%) (原始: 34.2%)")

# ============ 保存 ============
df['topic_id'] = new_topic_ids
df['topic_name'] = df['topic_id'].map(
    lambda x: new_keywords.get(x, 'Outlier')
)
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

topic_info = pd.DataFrame([
    {'Topic': c, 'Count': new_sizes[c], 'Keywords': new_keywords[c]}
    for c in sorted(new_sizes.keys())
])
topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False, encoding='utf-8-sig')

summary = {
    'method': 'LLM-assisted topic merging (MiniMax-M2.7)',
    'base_method': 'BERTopic aggressive (Experiment B)',
    'original_n_topics': len(valid_topics),
    'merged_n_topics': n_merged_topics,
    'n_merge_groups': len(merge_groups),
    'n_llm_evaluated': len(eval_candidates) if API_KEY else 0,
    'n_llm_merge': sum(1 for v in merge_decisions.values() if v) if merge_decisions else 0,
    'final_n_topics': n_merged_topics,
    'outlier_pct': float(new_outlier_pct),
    'merge_decisions': {f"{t1}-{t2}": v for (t1, t2), v in merge_decisions.items()},
}
with open(OUT_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"\n  ✅ 实验 J 完成!")
print(f"  📊 话题数: {len(valid_topics)} → {n_merged_topics} (LLM合并 {sum(1 for v in merge_decisions.values() if v) if merge_decisions else 0} 对)")
print(f"  📈 离群率: {new_outlier_pct:.1f}%")
print(f"  📁 {OUT_DIR}")
