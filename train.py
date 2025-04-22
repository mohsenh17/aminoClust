import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter
from models.amino_clust import VQVAE 

aminoacid_str = [
    "A", # Alanine
    "C", # Cysteine
    "D", # Aspartic acid
    "E", # Glutamic acid
    "F", # Phenylalanine
    "G", # Glycine
    "H", # Histidine
    "I", # Isoleucine
    "K", # Lysine
    "L", # Leucine
    "M", # Methionine
    "N", # Asparagine
    "P", # Proline
    "Q", # Glutamine
    "R", # Arginine
    "S", # Serine
    "T", # Threonine
    "V", # Valine
    "W", # Tryptophan
    "Y" # Tyrosine
]
aminoacid_dict = {aminoacid: i+1 for i, aminoacid in enumerate(aminoacid_str)}#.update({"X": 0})  # X for unknown
reversed_aminoacid_dict = {v: k for k, v in aminoacid_dict.items()}

def prepare_data(config):
    input_dim = config['model']['input_dim']
    data_dir = config['base']['train_dir']

    embeddings = []
    aas = []
    with open(data_dir, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            aa, vec = line.split(':')
            vec_values = list(map(float, vec.strip().split()))
            if len(vec_values) != input_dim:
                raise ValueError(f"Expected {input_dim} values, got {len(vec_values)}")
            embeddings.append(vec_values)
            aas.append(aminoacid_dict[aa])

    embeddings = torch.tensor(embeddings, dtype=torch.float32)  # (L, 1024)
    full_dataset = TensorDataset(embeddings, torch.tensor(aas))

    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len

    train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader

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
    writer = SummaryWriter(log_dir=os.path.join("tb_logs", model_name))

    train_loader, val_loader, _ = prepare_data(config)

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
        config['model']['name'] = f"aminoClust_{i}"
        #config['base']['checkpoint_dir'] = f"checkpoints/{config['model']['name']}"
        config['model']['num_clusters'] = i
        os.makedirs(config['base']['checkpoint_dir'], exist_ok=True)
        train_model(config)