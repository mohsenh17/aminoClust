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

if __name__ == "__main__":
    # Example usage
    data_dir = "data"
    dataset = ProteinDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in dataloader:
        embeddings, aa_sequence = batch  # embeddings: (1, L, 1024), aa_sequence: list of lists
        print(embeddings.shape, aa_sequence)
        break
