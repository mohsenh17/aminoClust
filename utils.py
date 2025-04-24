import os
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

    return embeddings, aas

def data_loader(config):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split
    embeddings, aas = prepare_data(config)
    
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
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

def list_files_and_write_contents(dir_path, output_file, max_files=10):
    # List all files and directories in the given path
    files = os.listdir(dir_path)
    # Filter out only files (not directories)
    files = [f for f in files if os.path.isfile(os.path.join(dir_path, f))][:max_files]
    
    # Open the output file in write mode
    with open(output_file, 'w') as out_file:
        for file in files:
            file_path = os.path.join(dir_path, file)
            
            # Read the contents of the file and write them into the output file
            with open(file_path, 'r') as in_file:
                content = in_file.read()
                out_file.write(content)
            
if __name__ == "__main__":
    # Example usage
    list_files_and_write_contents('/home/mohsenh/projects/def-ilie/mohsenh/ppi/dataset/embd/gold_std/', 'data/output_10.txt', max_files=10)
