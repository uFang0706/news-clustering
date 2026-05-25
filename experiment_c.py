"""
实验 C: 经典方案 - TF-IDF + K-Means
对比: BERTopic, 无监督确定K(肘部+轮廓), 人工评估语义一致性
"""

import warnings
warnings.filterwarnings("ignore")
import random, numpy as np, pandas as pd, re
from pathlib import Path
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# 确保 NLTK 资源
for resource in ['stopwords', 'punkt', 'wordnet', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_exp_c")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 加载
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
title_col = [c for c in df.columns if 'headline' in c.lower() or 'title' in c.lower()][0]
headlines = df[title_col].astype(str).tolist()
print(f"[{datetime.now().strftime('%H:%M:%S')}] 实验 C: TF-IDF + K-Means / LDA")
print(f"  标题数: {len(headlines):,}")

# ============ 文本预处理 ============
print("  文本预处理...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens
              if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)

processed = [preprocess(h) for h in headlines]

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000, min_df=3, max_df=0.85,
                         ngram_range=(1,2), sublinear_tf=True)
tfidf_matrix = tfidf.fit_transform(processed)
feature_names = tfidf.get_feature_names_out()
print(f"  TF-IDF 矩阵: {tfidf_matrix.shape}")

# ============ 选 K: 扫描 5~60 ============
print("  扫描最优 K (轮廓/Davies-Bouldin/Calinski-Harabasz)...")
K_RANGE = range(5, 61, 5)
scores = []
for k in K_RANGE:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
    labels = km.fit_predict(tfidf_matrix)
    sil = silhouette_score(tfidf_matrix, labels, sample_size=5000)
    db  = davies_bouldin_score(tfidf_matrix.toarray()[:5000], labels[:5000])
    ch  = calinski_harabasz_score(tfidf_matrix.toarray()[:5000], labels[:5000])
    scores.append({'K': k, 'silhouette': sil, 'davies_bouldin': db, 'calinski_harabasz': ch})
    print(f"    K={k:2d}: sil={sil:.4f}  db={db:.4f}  ch={ch:.0f}")

scores_df = pd.DataFrame(scores)
best_k_sil = scores_df.loc[scores_df['silhouette'].idxmax(), 'K']
best_k_db  = scores_df.loc[scores_df['davies_bouldin'].idxmin(), 'K']
best_k_ch  = scores_df.loc[scores_df['calinski_harabasz'].idxmax(), 'K']
print(f"\n  最优 K (轮廓): {best_k_sil} | (DB): {best_k_db} | (CH): {best_k_ch}")

# 综合取中位数
best_k = int(np.median([best_k_sil, best_k_db, best_k_ch]))
print(f"  综合最优 K = {best_k}")

# ============ 最终聚类 ============
print(f"\n  最终 K-Means (K={best_k})...")
km_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, n_init=10, batch_size=1024)
labels = km_final.fit_predict(tfidf_matrix)
df['kmeans_topic'] = labels

# 每类 top TF-IDF 关键词 → 话题命名
def top_tfidf_words(cluster_id, n=10):
    center = km_final.cluster_centers_[cluster_id]
    top_idx = center.argsort()[::-1][:n]
    return [feature_names[i] for i in top_idx]

topic_labels = {}
for c in range(best_k):
    words = top_tfidf_words(c)
    topic_labels[c] = f"Topic_{c}: " + ', '.join(words[:5])

df['topic_name'] = df['kmeans_topic'].map(topic_labels)

# 聚类分布
cluster_sizes = df['kmeans_topic'].value_counts().sort_values(ascending=False)
print(f"\n  Top 10 话题分布:")
for tid, size in cluster_sizes.head(10).items():
    print(f"    {tid}: {size} 条  →  {topic_labels[tid]}")

# ============ LDA (对比) ============
print(f"\n  LDA (Topic={best_k})...")
count_vec = CountVectorizer(max_features=5000, min_df=3, max_df=0.85,
                             stop_words='english', ngram_range=(1,2))
count_matrix = count_vec.fit_transform(processed)
lda = LatentDirichletAllocation(n_components=best_k, random_state=42,
                                 max_iter=10, learning_method='online',
                                 batch_size=1024)
lda_labels = lda.fit_transform(count_matrix)
lda_topic_assign = lda_labels.argmax(axis=1)
df['lda_topic'] = lda_topic_assign

def lda_top_words(model, feature_names, n=10):
    topics = {}
    for idx, topic in enumerate(model.components_):
        top = topic.argsort()[::-1][:n]
        topics[idx] = ', '.join([feature_names[i] for i in top])
    return topics

lda_topic_words = lda_top_words(lda, count_vec.get_feature_names_out())
print(f"  LDA 完成，话题分布:")
for tid in sorted(df['lda_topic'].value_counts().index)[:10]:
    size = (df['lda_topic']==tid).sum()
    print(f"    LDA_{tid}: {size} 条  →  {lda_topic_words[tid]}")

# ============ 保存结果 ============
scores_df.to_csv(OUT_DIR / "k_selection_scores.csv", index=False)
df.to_csv(OUT_DIR / "headlines_with_topics.csv", index=False, encoding='utf-8-sig')

# 可视化: K选择曲线
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(scores_df['K'], scores_df['silhouette'], 'o-', color='steelblue')
axes[0].axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
axes[0].set_title('Silhouette Score (higher=better)')
axes[0].set_xlabel('K'); axes[0].legend()

axes[1].plot(scores_df['K'], scores_df['davies_bouldin'], 'o-', color='orange')
axes[1].axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
axes[1].set_title('Davies-Bouldin Index (lower=better)')
axes[1].set_xlabel('K'); axes[1].legend()

axes[2].plot(scores_df['K'], scores_df['calinski_harabasz'], 'o-', color='green')
axes[2].axvline(best_k, color='red', linestyle='--', label=f'K={best_k}')
axes[2].set_title('Calinski-Harabasz Index (higher=better)')
axes[2].set_xlabel('K'); axes[2].legend()

plt.tight_layout()
plt.savefig(OUT_DIR / "k_selection.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  ✅ 实验 C 完成!")
print(f"  📁 {OUT_DIR}")
print(f"  最优 K={best_k}  |  轮廓系数={scores_df[scores_df['K']==best_k]['silhouette'].values[0]:.4f}")
