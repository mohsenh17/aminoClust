import torch
from torch.utils.data import Dataset, DataLoader
import os

MAX_LENGTH = 800
EMBED_DIM = 1024

class ProteinDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".embd")][:100]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        embeddings = []

        with open(file_path, 'r') as f:
            for line in f:
                _, vec = line.strip().split(':')
                vec_values = list(map(float, vec.strip().split()))
                if len(vec_values) != EMBED_DIM:
                    raise ValueError(f"Expected {EMBED_DIM} values, got {len(vec_values)} in {file_path}")
                embeddings.append(vec_values)

        embedding_tensor = torch.tensor(embeddings, dtype=torch.float32)  # (L, 1024)
        return embedding_tensor


def protein_collate_fn(batch):
    batch_padded = []
    for protein in batch:
        length = protein.shape[0]
        if length >= MAX_LENGTH:
            padded = protein[:MAX_LENGTH]  # truncate
        else:
            pad_size = MAX_LENGTH - length
            padding = torch.zeros((pad_size, EMBED_DIM))
            padded = torch.cat([protein, padding], dim=0)
        batch_padded.append(padded)

    return torch.stack(batch_padded)  # shape: (batch_size, 800, 1024)

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset = ProteinDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=protein_collate_fn)

    for batch in dataloader:
        print(batch.shape)  # torch.Size([32, 800, 1024])
        break
