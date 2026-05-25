#!/usr/bin/env python3
"""
系统对比5个方案 + 修复基线
数据集: ABC News 英文新闻标题 (sampled_headlines.csv)
评估指标: Silhouette, Calinski-Harabasz, Davies-Bouldin, 话题一致性
"""

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from collections import Counter, defaultdict

# ============ 配置 ============
OUT_DIR = Path("/Users/yuwan/code/news-clustering/output_benchmark")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLUSTERS = 50  # 统一簇数，公平对比
SAMPLE_SIZE = 10000  # 先用10000条加速，最后跑全量

# ============ 加载数据 ============
print(f"[{datetime.now().strftime('%H:%M:%S')}] 加载数据...")
df = pd.read_csv("/Users/yuwan/code/news-clustering/output/sampled_headlines.csv")
headlines = df['headline_text'].astype(str).tolist()[:SAMPLE_SIZE]
print(f"  标题数: {len(headlines):,}")

# ============ 评估函数 ============
def evaluate_clustering(X, labels, method_name, headlines=None):
    """统一评估指标"""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    # 过滤离群点
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]
    
    if len(set(labels_valid)) < 2:
        return {
            'method': method_name,
            'silhouette': float('nan'),
            'calinski_harabasz': float('nan'),
            'davies_bouldin': float('nan'),
            'n_clusters': len(set(labels)),
            'coverage': mask.sum() / len(labels),
        }
    
    sil = silhouette_score(X_valid, labels_valid, metric='euclidean')
    ch = calinski_harabasz_score(X_valid, labels_valid)
    db = davies_bouldin_score(X_valid, labels_valid)
    
    result = {
        'method': method_name,
        'silhouette': float(sil),
        'calinski_harabasz': float(ch),
        'davies_bouldin': float(db),
        'n_clusters': len(set(labels_valid)),
        'coverage': mask.sum() / len(labels),
    }
    
    # 话题一致性 (Topic Coherence)
    if headlines is not None:
        coherence = compute_topic_coherence(headlines, labels)
        result['topic_coherence'] = coherence
    
    return result

def compute_topic_coherence(headlines, labels, top_n=10):
    """计算话题一致性: 每个簇内高频词的共现程度"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # 过滤离群点
    mask = labels != -1
    valid_headlines = [headlines[i] for i in range(len(headlines)) if mask[i]]
    valid_labels = labels[mask]
    
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
    tfidf_matrix = tfidf.fit_transform(valid_headlines)
    feature_names = tfidf.get_feature_names_out()
    
    coherences = []
    for c in set(valid_labels):
        cluster_mask = valid_labels == c
        if cluster_mask.sum() < 5:
            continue
        
        # 取簇内Top词
        mean_vec = tfidf_matrix[cluster_mask].mean(axis=0).A1
        top_idx = mean_vec.argsort()[::-1][:top_n]
        top_words = [feature_names[i] for i in top_idx]
        
        # 计算共现分数 (简化版)
        score = mean_vec[top_idx].mean()
        coherences.append(score)
    
    return np.mean(coherences) if coherences else 0.0

# ============ 基线方法 (修复版) ============
print(f"\n{'='*70}")
print("基线方法: SBERT + PCA + GMM (修复版)")
print(f"{'='*70}")

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(headlines, show_progress_bar=True, batch_size=256)
print(f"  Embedding: {embeddings.shape}")

# FIX: 不用StandardScaler，直接PCA（避免数值溢出）
pca = PCA(n_components=50, random_state=42)
X_pca = pca.fit_transform(embeddings)
print(f"  PCA: {X_pca.shape}, 解释方差={pca.explained_variance_ratio_.sum()*100:.1f}%")

gmm_pca = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag', 
                          random_state=42, n_init=3, max_iter=200)
labels_pca = gmm_pca.fit_predict(X_pca)

result_pca = evaluate_clustering(X_pca, labels_pca, "Baseline: PCA+GMM(diag)", headlines)
print(f"  Silhouette: {result_pca['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_pca['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_pca['davies_bouldin']:.4f}")

results = [result_pca]

# ============ 方案一: 升级Embedding + 优化GMM ============
print(f"\n{'='*70}")
print("方案一: all-mpnet-base-v2 + PCA(200) + GMM(full)")
print(f"{'='*70}")

model_mpnet = SentenceTransformer('all-mpnet-base-v2')
embeddings_mpnet = model_mpnet.encode(headlines, show_progress_bar=True, batch_size=128)

n_comp = min(200, embeddings_mpnet.shape[0] - 1)
pca200 = PCA(n_components=n_comp)
X_pca200 = pca200.fit_transform(embeddings_mpnet)

gmm_full = GaussianMixture(n_components=N_CLUSTERS, covariance_type='full',
                           random_state=42, n_init=3, max_iter=200, reg_covar=1e-3)
labels_mpnet = gmm_full.fit_predict(X_pca200)

result_mpnet = evaluate_clustering(X_pca200, labels_mpnet, "方案一: mpnet+PCA200+GMM(full)", headlines)
print(f"  Silhouette: {result_mpnet['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_mpnet['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_mpnet['davies_bouldin']:.4f}")

results.append(result_mpnet)

# ============ 方案二: GSDMM ============
print(f"\n{'='*70}")
print("方案二: GSDMM (短文本专用)")
print(f"{'='*70}")

class GSDMM:
    def __init__(self, K=50, alpha=0.1, beta=0.1, n_iters=30):
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters
        
    def fit(self, texts):
        all_words = set()
        tokenized = []
        for text in texts:
            words = text.lower().split()
            tokenized.append(words)
            all_words.update(words)
        
        vocab = list(all_words)
        word2id = {w: i for i, w in enumerate(vocab)}
        V = len(vocab)
        D = len(texts)
        
        self.labels = np.random.randint(0, self.K, size=D)
        self.cluster_doc_count = np.zeros(self.K, dtype=int)
        self.cluster_word_count = np.zeros(self.K, dtype=int)
        self.cluster_word_freq = np.zeros((self.K, V), dtype=int)
        
        doc_words = []
        for words in tokenized:
            wids = [word2id[w] for w in words if w in word2id]
            doc_words.append(wids)
        
        for d, (words, z) in enumerate(zip(doc_words, self.labels)):
            self.cluster_doc_count[z] += 1
            self.cluster_word_count[z] += len(words)
            for w in words:
                self.cluster_word_freq[z, w] += 1
        
        for it in range(self.n_iters):
            changes = 0
            for d, words in enumerate(doc_words):
                if len(words) == 0:
                    continue
                z_old = self.labels[d]
                self.cluster_doc_count[z_old] -= 1
                self.cluster_word_count[z_old] -= len(words)
                for w in words:
                    self.cluster_word_freq[z_old, w] -= 1
                
                scores = np.zeros(self.K)
                for k in range(self.K):
                    score = np.log(self.cluster_doc_count[k] + self.alpha)
                    for w in words:
                        score += np.log(self.cluster_word_freq[k, w] + self.beta)
                        score -= np.log(self.cluster_word_count[k] + V * self.beta)
                    scores[k] = score
                
                scores = scores - np.max(scores)
                probs = np.exp(scores)
                probs = probs / np.sum(probs)
                z_new = np.random.choice(self.K, p=probs)
                
                self.labels[d] = z_new
                self.cluster_doc_count[z_new] += 1
                self.cluster_word_count[z_new] += len(words)
                for w in words:
                    self.cluster_word_freq[z_new, w] += 1
                
                if z_new != z_old:
                    changes += 1
        
        return self

gsdmm = GSDMM(K=N_CLUSTERS, alpha=0.1, beta=0.1, n_iters=30)
gsdmm.fit(headlines)
labels_gsdmm = gsdmm.labels

# GSDMM没有向量空间，用TF-IDF近似评估
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=500, stop_words='english')
X_tfidf = tfidf.fit_transform(headlines).toarray()

result_gsdmm = evaluate_clustering(X_tfidf, labels_gsdmm, "方案二: GSDMM", headlines)
print(f"  Silhouette: {result_gsdmm['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_gsdmm['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_gsdmm['davies_bouldin']:.4f}")

results.append(result_gsdmm)

# ============ 方案三: TopiCLEAR (简化实现) ============
print(f"\n{'='*70}")
print("方案三: TopiCLEAR (LDA自适应投影)")
print(f"{'='*70}")

# TopiCLEAR核心: GMM初步聚类 → LDA投影优化 → 迭代至收敛
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: GMM初步聚类
gmm_init = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag',
                           random_state=42, n_init=3)
labels_init = gmm_init.fit_predict(X_pca)

# Step 2: LDA监督投影 (迭代优化)
X_topi = X_pca.copy()
labels_topi = labels_init.copy()

for iteration in range(5):  # 迭代5次
    # LDA投影
    lda = LinearDiscriminantAnalysis(n_components=min(N_CLUSTERS-1, X_topi.shape[1]))
    try:
        X_lda = lda.fit_transform(X_topi, labels_topi)
    except:
        break  # LDA失败则停止
    
    # GMM重新聚类
    gmm_iter = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag',
                               random_state=42, n_init=1)
    labels_new = gmm_iter.fit_predict(X_lda)
    
    # 检查收敛
    if np.array_equal(labels_new, labels_topi):
        print(f"  收敛于迭代 {iteration}")
        break
    labels_topi = labels_new
    X_topi = X_lda

result_topi = evaluate_clustering(X_topi, labels_topi, "方案三: TopiCLEAR", headlines)
print(f"  Silhouette: {result_topi['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_topi['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_topi['davies_bouldin']:.4f}")

results.append(result_topi)

# ============ 方案四: UMAP优化 ============
print(f"\n{'='*70}")
print("方案四: UMAP降维 + GMM")
print(f"{'='*70}")

from umap import UMAP
umap_reducer = UMAP(n_components=50, metric='cosine', random_state=42, 
                    min_dist=0.1, n_neighbors=15)
X_umap = umap_reducer.fit_transform(embeddings)

gmm_umap = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag',
                           random_state=42, n_init=3)
labels_umap = gmm_umap.fit_predict(X_umap)

result_umap = evaluate_clustering(X_umap, labels_umap, "方案四: UMAP+GMM", headlines)
print(f"  Silhouette: {result_umap['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_umap['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_umap['davies_bouldin']:.4f}")

results.append(result_umap)

# ============ 方案五: 完整Pipeline ============
print(f"\n{'='*70}")
print("方案五: 完整Pipeline (英文版)")
print(f"{'='*70}")

# 1. 预处理: 小写 + 去停用词 (简单版)
import re
processed = []
for h in headlines:
    h = h.lower()
    h = re.sub(r'[^a-z\s]', '', h)
    processed.append(h)

# 2. TF-IDF + Embedding融合
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_features=1000, stop_words='english', min_df=2)
X_tfidf_feat = tfidf_vec.fit_transform(processed).toarray()

# 拼接Embedding和TF-IDF特征
X_combined = np.hstack([embeddings, X_tfidf_feat])
print(f"  融合特征维度: {X_combined.shape}")

# 3. PCA降维
pca_pipe = PCA(n_components=50)
X_pipe = pca_pipe.fit_transform(X_combined)

# 4. GMM聚类
gmm_pipe = GaussianMixture(n_components=N_CLUSTERS, covariance_type='diag',
                           random_state=42, n_init=3)
labels_pipe = gmm_pipe.fit_predict(X_pipe)

result_pipe = evaluate_clustering(X_pipe, labels_pipe, "方案五: Pipeline", headlines)
print(f"  Silhouette: {result_pipe['silhouette']:.4f}")
print(f"  Calinski-Harabasz: {result_pipe['calinski_harabasz']:.1f}")
print(f"  Davies-Bouldin: {result_pipe['davies_bouldin']:.4f}")

results.append(result_pipe)

# ============ 汇总对比 ============
print(f"\n{'='*70}")
print("汇总对比")
print(f"{'='*70}")

df_results = pd.DataFrame(results)
cols = ['method', 'silhouette', 'calinski_harabasz', 'davies_bouldin', 'n_clusters', 'coverage']
print(df_results[cols].to_string(index=False))

# 保存
with open(OUT_DIR / "benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ 完成! 结果保存: {OUT_DIR}")
