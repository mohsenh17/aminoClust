import sys
sys.path.insert(0, '.')
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
from evaluation.dimension_reduction import process_latents


def evaluate_clusters(embeddings, cluster_labels):
    """
    Evaluate clustering quality using intrinsic metrics.

    Args:
        embeddings (np.ndarray or torch.Tensor): shape (N, D)
        cluster_labels (np.ndarray or list): shape (N,)

    Returns:
        dict: dictionary of scores
    """
    if hasattr(embeddings, 'detach'):
        embeddings = embeddings.detach().cpu().numpy()
    if hasattr(cluster_labels, 'detach'):
        cluster_labels = cluster_labels.detach().cpu().numpy()

    results = {}
    results['silhouette_score'] = silhouette_score(embeddings, cluster_labels)
    results['davies_bouldin_score'] = davies_bouldin_score(embeddings, cluster_labels)
    results['calinski_harabasz_score'] = calinski_harabasz_score(embeddings, cluster_labels)

    return results

if __name__ == "__main__":
    import yaml
    import os
    import torch
    from models.amino_clust import VQVAE
    from train import prepare_data, reversed_aminoacid_dict
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in [2,4,8,16,32,64]:
        config['model']['latent_dim'] = j
        for i in range (2,21):
            config['model']['name'] = f"aminoClust_{config['model']['latent_dim']}_{i}"
            config['model']['num_clusters'] = i
            checkpoint_path = config['base']['checkpoint_dir'] + f"/{config['model']['name']}.pth"
            checkpoint = torch.load(checkpoint_path, map_location=device)
            output_file = config['base']['evaluation_dir'] + f"/tsne_latents_{config['model']['name']}.csv"

            model = VQVAE(config['model']['input_dim'],config['model']['latent_dim'], config['model']['num_clusters']).to(device)
            train_loader, val_loader, test_loader = prepare_data(config)
            latents, cluster_ids, labels = process_latents(model, test_loader, device)
            results = evaluate_clusters(latents, cluster_ids)
            print("=====================================")
            print(f"Model: {config['model']['name']}")
            print(f"Evaluation results for {config['model']['name']}:")
            print(f"Silhouette Score: {results['silhouette_score']}")
            print(f"Davies-Bouldin Score: {results['davies_bouldin_score']}")
            print(f"Calinski-Harabasz Score: {results['calinski_harabasz_score']}")
            print("=====================================")
            
            model.load_state_dict(checkpoint)