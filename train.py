import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
#from models.amino_clust import VQVAE 
from models.amino_clust_dense import VQVAE 


def train_model(config):
    latent_dim = config['model']['latent_dim']
    num_embeddings = config['model']['num_clusters']
    input_dim = config['model']['input_dim']
    learning_rate = config['model']['learning_rate']
    epochs = config['model']['num_epochs']
    model_name = config['model']['name']
    checkpoint_dir = config['base']['checkpoint_dir']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VQVAE(input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    writer = SummaryWriter(log_dir=os.path.join(config['base']['log_dir'], model_name))

    train_loader, val_loader, _ = data_loader(config)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss, total_recon, total_vq = 0.0, 0.0, 0.0
        total_unique_codes = set()

        for batch in train_loader:
            batch = batch[0].to(device)
            optimizer.zero_grad()
            x_recon, vq_loss, encoding_indices = model(batch)
            recon_loss = criterion(x_recon, batch)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_vq += vq_loss.item()
            total_unique_codes.update(torch.unique(encoding_indices).tolist())

        # === Validation ===
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = val_batch[0].to(device)
                x_recon, vq_loss, _ = model(val_batch)
                recon_loss = criterion(x_recon, val_batch)
                val_loss_total += (recon_loss + vq_loss).item()

        avg_val_loss = val_loss_total / len(val_loader)

        # Logging
        writer.add_scalar('Loss/train_total', total_loss, epoch)
        writer.add_scalar('Loss/train_recon', total_recon, epoch)
        writer.add_scalar('Loss/train_vq', total_vq, epoch)
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch)
        writer.add_scalar('Metrics/unique_codes_used', len(total_unique_codes), epoch)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{checkpoint_dir}/{model_name}.pth")
            saved = "(saved)"
        else:
            saved = ""

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {total_loss:.4f}, Recon: {total_recon:.4f}, VQ: {total_vq:.4f}, "
              f"Unique Codes: {len(total_unique_codes)} / {num_embeddings}, "
              f"Val Loss: {avg_val_loss:.4f} {saved}")

    writer.close()


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    for i in range (2,21):
        config['model']['name'] = f"aminoClust_dense_{config['model']['latent_dim']}_{i}"
        config['model']['num_clusters'] = i
        if os.path.exists(config['base']['checkpoint_dir'] + f"/{config['model']['name']}.pth"):
            print(f'checkpoint for {config["model"]["name"]} exist!')
            continue
        os.makedirs(config['base']['checkpoint_dir'], exist_ok=True)
        train_model(config)