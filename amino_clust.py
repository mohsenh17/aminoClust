import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from dataloader import (ProteinDataset, protein_collate_fn,
                        AminoAcidDataset)


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



latent_dim = 100  # latent space dimension
num_embeddings = 21  # each embedding can represent a unique cluster or cell type
input_dim = 1024
model = VQVAE(input_dim=input_dim, latent_dim=latent_dim, num_embeddings=num_embeddings)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
