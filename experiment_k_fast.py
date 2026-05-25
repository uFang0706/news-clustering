"""
Experiment K: Consensus Clustering (Fast Version)
Use sampling + efficient co-association computation
"""
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
import json
import os

print("Loading clustering results...")

# Load all labels
gmm_df = pd.read_csv('output_exp_i2/headlines_with_topics.csv')
leiden_df = pd.read_csv('output_exp_i/headlines_with_topics.csv')
bert_df = pd.read_csv('output_exp_b/headlines_with_topics.csv')

gmm_labels = gmm_df['topic_id'].values
leiden_labels = leiden_df['topic_id'].values
bert_labels = bert_df['topic_id'].values

print(f"GMM: {len(np.unique(gmm_labels))} clusters")
print(f"Leiden: {len(np.unique(leiden_labels))} clusters")
print(f"BERTopic: {len(np.unique(bert_labels))} clusters (incl. -1)")

n_samples = len(gmm_labels)

# Fast co-association: use label equality vectors
print("Building co-association matrix efficiently...")

# Sample for faster computation
sample_size = 10000
np.random.seed(42)
idx = np.random.choice(n_samples, sample_size, replace=False)
idx.sort()

gmm_s = gmm_labels[idx]
leiden_s = leiden_labels[idx]
bert_s = bert_labels[idx]

# Build co-association for sample
coassoc = np.zeros((sample_size, sample_size), dtype=np.float32)

# Vectorized pairwise comparisons
for i in range(sample_size):
    gmm_match = (gmm_s == gmm_s[i])
    leiden_match = (leiden_s == leiden_s[i])
    bert_match = (bert_s == bert_s[i]) & (bert_s != -1) & (bert_s[i] != -1)
    
    coassoc[i, :] = gmm_match.astype(np.float32) + leiden_match.astype(np.float32) + bert_match.astype(np.float32)

coassoc = coassoc / 3.0  # normalize
coassoc[np.arange(sample_size), np.arange(sample_size)] = 0

distance = 1 - coassoc

print(f"Distance matrix: {distance.shape}")

# Try different cluster counts
results = []
for n_clusters in [20, 30, 40, 50, 60, 80, 100]:
    print(f"\nConsensus n_clusters={n_clusters}...")
    
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance)
    
    sizes = Counter(labels)
    result = {
        'n_clusters': n_clusters,
        'n_unique': len(np.unique(labels)),
        'min_size': int(min(sizes.values())),
        'max_size': int(max(sizes.values())),
        'median_size': float(np.median(list(sizes.values()))),
    }
    results.append(result)
    print(f"  Unique: {result['n_unique']}, sizes: {result['min_size']}-{result['max_size']}, median: {result['median_size']:.0f}")

# Save
os.makedirs('output_exp_k', exist_ok=True)
with open('output_exp_k/summary.json', 'w') as f:
    json.dump({
        'method': 'Consensus Clustering (sampled)',
        'sample_size': sample_size,
        'algorithms': ['GMM', 'Leiden', 'BERTopic'],
        'scan_results': results
    }, f, indent=2)

# Best: choose based on median size being reasonable (~500-1000)
best = min(results, key=lambda x: abs(x['median_size'] - 800))
print(f"\nBest (by median size ~800): n_clusters={best['n_clusters']}")

# Re-run on full data with best n_clusters
print("\nRunning on full dataset...")
# For full data, use a simpler approach: majority vote per sample
# Map each sample's (gmm, leiden, bert) tuple to consensus

# Use the sample to build a mapping, then apply to full
clustering = AgglomerativeClustering(
    n_clusters=best['n_clusters'],
    metric='precomputed',
    linkage='average'
)
labels_sample = clustering.fit_predict(distance)

# Create mapping from algorithm labels to consensus
# For each consensus cluster, find dominant algorithm labels
from collections import defaultdict

consensus_map = {}
for c in np.unique(labels_sample):
    mask = labels_sample == c
    gmm_mode = Counter(gmm_s[mask]).most_common(1)[0][0]
    leiden_mode = Counter(leiden_s[mask]).most_common(1)[0][0]
    consensus_map[(gmm_mode, leiden_mode)] = c

# Assign full data using nearest match
full_labels = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    key = (gmm_labels[i], leiden_labels[i])
    if key in consensus_map:
        full_labels[i] = consensus_map[key]
    else:
        # Find nearest key
        best_key = min(consensus_map.keys(), 
                      key=lambda k: (k[0]!=key[0]) + (k[1]!=key[1]))
        full_labels[i] = consensus_map[best_key]

# Save
sizes = Counter(full_labels)
print(f"Full consensus: {len(np.unique(full_labels))} clusters")
print(f"Sizes: min={min(sizes.values())}, max={max(sizes.values())}, median={np.median(list(sizes.values())):.0f}")

df_out = pd.DataFrame({
    'publish_date': gmm_df['publish_date'],
    'headline_text': gmm_df['headline_text'],
    'consensus_topic': full_labels,
    'gmm_topic': gmm_labels,
    'leiden_topic': leiden_labels,
    'bertopic_topic': bert_labels
})
df_out.to_csv('output_exp_k/headlines_with_topics.csv', index=False)

print("\nExperiment K complete!")
