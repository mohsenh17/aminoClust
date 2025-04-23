import sys
sys.path.insert(0, '.')
from sklearn.manifold import TSNE
import torch
from models.amino_clust import VQVAE 
from train import prepare_data, reversed_aminoacid_dict
import pandas as pd


def process_latents(model, dataloader, device, n_samples=3):
    model.eval()
    latents = []
    cluster_ids = []
    labels = []

    with torch.no_grad():
        for i, (batch, label) in enumerate(dataloader):
            if len(latents) >= n_samples:
                break
            batch = batch.to(device)
            z_e = model.encoder(batch)  # shape: (B, D)
            _, _, encoding_indices = model.vq(z_e)
            latents.append(z_e.cpu())
            cluster_ids.append(encoding_indices.view(-1).cpu())
            labels.append(label.cpu())

    latents = torch.cat(latents, dim=0)
    cluster_ids = torch.cat(cluster_ids, dim=0)
    labels = torch.cat(labels, dim=0)

    return latents, cluster_ids, labels

def apply_tsne_and_create_dataframe(latents, cluster_ids, labels, output_file):
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    latents_2d = tsne.fit_transform(latents.numpy())

    df = pd.DataFrame(latents_2d, columns=['x', 'y'])
    df['cluster_id'] = cluster_ids.numpy()
    df['label'] = pd.Series(labels).map(reversed_aminoacid_dict)
    df.to_csv(output_file, index=False)

    return df

if __name__ == "__main__":
    import yaml
    import os
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for j in [2,4,8,16,32,64]:
        config['model']['latent_dim'] = j
        for i in range (2,21):
            config['model']['name'] = f"aminoClust_{config['model']['latent_dim']}_{i}"
            config['model']['num_clusters'] = i
            checkpoint_path = config['base']['checkpoint_dir'] + f"/{config['model']['name']}.pth"
            if os.path.exists(checkpoint_path):
                print(f"checkpoint for {config['model']['name']} exist!")
                continue
            checkpoint = torch.load(checkpoint_path, map_location=device)
            output_file = config['base']['evaluation_dir'] + f"/tsne_latents_{config['model']['name']}.csv"
            os.makedirs(config['base']['evaluation_dir'], exist_ok=True)

            model = VQVAE(config['model']['input_dim'],config['model']['latent_dim'], config['model']['num_clusters']).to(device)
            train_loader, val_loader, test_loader = prepare_data(config)

            model.load_state_dict(checkpoint)
            latents, cluster_ids, labels = process_latents(model, test_loader, device, n_samples=1000)
            apply_tsne_and_create_dataframe(latents, cluster_ids, labels, output_file)

