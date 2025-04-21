import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import yaml

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.1):
        """
        Args:
            num_embeddings (int): Number of vectors in the codebook.
            embedding_dim (int): Dimensionality of each codebook vector.
            commitment_cost (float): Weighting factor for commitment loss.
        """
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e):
        """
        Args:
            z_e (Tensor): Encoder output, shape (batch_size, latent_dim)
        Returns:
            z_q (Tensor): Quantized output, same shape as z_e.
            loss (Tensor): The VQ loss which includes the commitment loss.
            encoding_indices (Tensor): The codebook index for each latent vector.
        """
        distances = (torch.sum(z_e ** 2, dim=-1, keepdim=True) +
                     torch.sum(self.embeddings.weight ** 2, dim=1).unsqueeze(0) -
                     2 * torch.matmul(z_e, self.embeddings.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=-1, keepdim=True)
        
        z_q = self.embeddings(encoding_indices).view(z_e.shape)
        
    
        loss = F.mse_loss(z_q.detach(), z_e) + self.commitment_cost * F.mse_loss(z_e.detach(), z_q)
        
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, loss, encoding_indices

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Args:
            input_dim (int): Dimensionality of input data (e.g., number of HVGs).
            latent_dim (int): Dimensionality of the latent representation.
        """
        super(Encoder, self).__init__()
        self.linear = nn.Linear(input_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input data, shape (batch_size, input_dim)
        Returns:
            Tensor: Latent representation of shape (batch_size, latent_dim)
        """
        return self.linear(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        """
        Args:
            latent_dim (int): Dimensionality of the latent representation.
            output_dim (int): Dimensionality of the reconstructed output (same as input_dim).
        """
        super(Decoder, self).__init__()
        self.linear = nn.Linear(latent_dim, output_dim)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Latent code, shape (batch_size, latent_dim)
        Returns:
            Tensor: Reconstructed input, shape (batch_size, output_dim)
        """
        return self.linear(x)


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, num_embeddings):
        """
        Args:
            input_dim (int): Dimensionality of the input (e.g., number of HVGs).
            latent_dim (int): Dimensionality of the latent space.
            num_embeddings (int): Number of discrete latent codes in the codebook.
        """
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
    
    def forward(self, x):
        """
        Forward pass through the VQ-VAE.
        
        Args:
            x (Tensor): Input data, shape (batch_size, input_dim)
        Returns:
            x_recon (Tensor): Reconstructed input.
            vq_loss (Tensor): Vector quantization loss.
            encodings (Tensor): Discrete code indices representing clusters.
        """
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, encoding_indices
    
    def predict(self, x):
        """
        Given input data, output the discrete cluster assignments (codebook indices).
        
        Args:
            x (Tensor): Input data.
        Returns:
            encoding_indices (Tensor): Discrete code indices.
        """
        self.eval()
        with torch.no_grad():
            x = x.to(next(self.parameters()).device)
            z_e = self.encoder(x)
            z_q, _, encoding_indices = self.vq(z_e)
        return encoding_indices



latent_dim = config['model']['latent_dim']  # latent space dimension
num_embeddings = config['model']['num_clusters']  # each embedding can represent a unique cluster or cell type
input_dim = config['model']['input_dim']
model = VQVAE(input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings)
optimizer = torch.optim.Adam(model.parameters(), lr=config['model']['learning_rate'])
criterion = nn.MSELoss()


model.train()
epochs = config['model']['num_epochs']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

embeddings = []
data_dir = config['base']['train_dir']
for line in open(data_dir, 'r'):
    line = line.strip()
    if not line:
        continue
    aa, vec = line.split(':')
    vec_values = list(map(float, vec.strip().split()))
    if len(vec_values) != input_dim:
        raise ValueError(f"Expected {input_dim} values, got {len(vec_values)}")
    # Process the vector as needed
    # For example, you can convert it to a tensor and store it in a list
    embeddings.append(vec_values)
embeddings = torch.tensor(embeddings, dtype=torch.float32)  # (L, 1024)
# Create a dataset and dataloader

full_dataset = TensorDataset(embeddings)

total_len = len(full_dataset)
train_len = int(0.8 * total_len)
val_len = int(0.1 * total_len)
test_len = total_len - train_len - val_len

train_ds, val_ds, test_ds = random_split(full_dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
model.train()
epochs = 100
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

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f"{config['base']['checkpoint_dir']}/{config['model']['name']}.pth")
        saved = "(saved)"
    else:
        saved = ""

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {total_loss:.4f}, Recon: {total_recon:.4f}, VQ: {total_vq:.4f}, "
          f"Unique Codes: {len(total_unique_codes)} / {num_embeddings}, "
          f"Val Loss: {avg_val_loss:.4f} {saved}")
