import sys
sys.path.insert(0, '.')
import os
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from evaluation.dimension_reduction import apply_tsne_and_create_dataframe
from utils import prepare_data
import yaml

with open("configs/config_kmeans.yaml", 'r') as f:
    config = yaml.safe_load(f)

from sklearn.cluster import KMeans
import numpy as np

def run_kmeans_pipeline(config, num_clusters=3, output_file=config['base']['evaluation_dir'] + "/kmeans_results.csv"):
    # Prepare data
    embeddings, aas = prepare_data(config=config)
    os.makedirs(config['base']['evaluation_dir'], exist_ok=True)

    # Run KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    # Apply t-SNE and save results
    apply_tsne_and_create_dataframe(
        latents=embeddings,
        cluster_ids=labels,
        labels=np.array(aas),
        output_file=output_file
    )

if __name__ == "__main__":
    # Example usage
    num_clusters = 3  # Set the desired number of clusters
    run_kmeans_pipeline(config, num_clusters=num_clusters)