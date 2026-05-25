"""
Experiment K: Consensus Clustering
Combine multiple clustering results via ensemble approach
"""
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from collections import Counter
import json
import os

print("Loading clustering results from experiments...")

# Load GMM labels (I.2 - our best)
gmm_df = pd.read_csv('output_exp_i2/headlines_with_topics.csv')
print(f"GMM columns: {gmm_df.columns.tolist()}")
gmm_labels = gmm_df['topic'].values if 'topic' in gmm_df.columns else gmm_df.iloc[:, -1].values

# Load Leiden labels (I)
leiden_df = pd.read_csv('output_exp_i/headlines_with_topics.csv')
print(f"Leiden columns: {leiden_df.columns.tolist()}")
leiden_labels = leiden_df['topic'].values if 'topic' in leiden_df.columns else leiden_df.iloc[:, -1].values

# Load BERTopic labels (B)
bert_df = pd.read_csv('output_exp_b/headlines_with_topics.csv')
print(f"BERTopic columns: {bert_df.columns.tolist()}")
bert_labels = bert_df['topic_id'].values

print(f"GMM labels: {len(gmm_labels)} samples, {len(np.unique(gmm_labels))} clusters")
print(f"Leiden labels: {len(leiden_labels)} samples, {len(np.unique(leiden_labels))} clusters")
print(f"BERTopic labels: {len(bert_labels)} samples, {len(np.unique(bert_labels))} clusters (incl. -1)")

# Build co-association matrix using sparse approach
n_samples = len(gmm_labels)
print(f"Building co-association matrix for {n_samples} samples...")

# Use a more efficient approach - count agreements per sample pair
from scipy.sparse import csr_matrix

def labels_to_membership(labels):
    """Convert labels to binary membership matrix"""
    n = len(labels)
    unique = np.unique(labels)
    membership = np.zeros((n, len(unique)), dtype=np.int8)
    for i, u in enumerate(unique):
        membership[labels == u, i] = 1
    return membership

gmm_mem = labels_to_membership(gmm_labels)
leiden_mem = labels_to_membership(leiden_labels)

# For BERTopic, handle -1 outliers
bert_valid = bert_labels != -1
bert_labels_valid = bert_labels[bert_valid]
bert_mem = np.zeros((n_samples, len(np.unique(bert_labels_valid))), dtype=np.int8)
for i, u in enumerate(np.unique(bert_labels_valid)):
    bert_mem[:, i] = ((bert_labels == u) & bert_valid).astype(np.int8)

# Compute co-association as dot products
coassoc = (gmm_mem @ gmm_mem.T + leiden_mem @ leiden_mem.T + bert_mem @ bert_mem.T)
coassoc = coassoc.toarray() if hasattr(coassoc, 'toarray') else np.array(coassoc)

# Zero diagonal
coassoc[np.arange(n_samples), np.arange(n_samples)] = 0

print(f"Co-association matrix built: {coassoc.shape}")
print(f"Agreement levels: min={coassoc.min()}, max={coassoc.max()}")
print(f"Mean agreement: {coassoc.mean():.4f}")

# Convert to distance matrix
distance = 1 - (coassoc / 3.0)

# Try different numbers of consensus clusters
results = []
for n_clusters in [20, 30, 40, 50, 60, 80, 100, 150]:
    print(f"\nTrying n_clusters={n_clusters}...")
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    
    labels = clustering.fit_predict(distance)
    
    n_outliers = len(labels[labels == -1]) if -1 in labels else 0
    coverage = (len(labels) - n_outliers) / len(labels) * 100
    
    # Calculate silhouette on a sample
    sample_idx = np.random.choice(n_samples, min(3000, n_samples), replace=False)
    try:
        sil = silhouette_score(distance[np.ix_(sample_idx, sample_idx)], labels[sample_idx], metric='precomputed')
    except Exception as e:
        sil = None
        print(f"  Silhouette error: {e}")
    
    sizes = Counter(labels)
    min_size = min(sizes.values())
    max_size = max(sizes.values())
    median_size = np.median(list(sizes.values()))
    
    result = {
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'outlier_pct': n_outliers / len(labels) * 100,
        'coverage': coverage,
        'silhouette': float(sil) if sil is not None else None,
        'min_size': int(min_size),
        'max_size': int(max_size),
        'median_size': float(median_size),
        'n_unique': int(len(np.unique(labels)))
    }
    results.append(result)
    print(f"  Clusters: {result['n_unique']}, Outliers: {result['outlier_pct']:.2f}%, Sil: {sil}")

# Save results
os.makedirs('output_exp_k', exist_ok=True)

with open('output_exp_k/summary.json', 'w') as f:
    json.dump({
        'method': 'Consensus Clustering (GMM + Leiden + BERTopic)',
        'n_algorithms': 3,
        'scan_results': results,
        'best_n_clusters': min(results, key=lambda x: abs(x['silhouette']) if x['silhouette'] is not None else 999)['n_clusters']
    }, f, indent=2)

# Save best result
best = min(results, key=lambda x: abs(x['silhouette']) if x['silhouette'] is not None else 999)
best_n = best['n_clusters']
print(f"\nBest: n_clusters={best_n}, silhouette={best['silhouette']:.4f}")

clustering = AgglomerativeClustering(
    n_clusters=best_n,
    metric='precomputed',
    linkage='average'
)
final_labels = clustering.fit_predict(distance)

# Save to CSV
df_out = pd.DataFrame({
    'headline_text': gmm_df['headline_text'] if 'headline_text' in gmm_df.columns else gmm_df['headline'],
    'consensus_topic': final_labels,
    'gmm_topic': gmm_labels,
    'leiden_topic': leiden_labels,
    'bertopic_topic': bert_labels
})
df_out.to_csv('output_exp_k/headlines_with_topics.csv', index=False)

print("\nExperiment K complete!")
print(f"Output: output_exp_k/")
