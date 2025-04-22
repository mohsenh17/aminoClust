# aminoClust  
🧬 **VQ-VAE Clustering of Amino Acids using Transformer-Based Embeddings**

This project explores unsupervised clustering of amino acids using **Vector Quantized Variational Autoencoders (VQ-VAE)** on top of embeddings extracted from transformer-based protein language models. By leveraging powerful protein representations, this work aims to uncover meaningful groupings of amino acids based on learned biochemical and structural properties.

---

## 🚀 Features
- **Embedding extraction** from transformer-based protein embedders (e.g., **ESM**, **ProtBert**, etc.)
- **Discrete latent space modeling** using **VQ-VAE**
- **Unsupervised clustering** of amino acids based on learned codebooks
- **Visualizations** of embedding spaces and cluster assignments
- Modular and extensible codebase for experimenting with different embedders and model architectures

---

## 🧠 Motivation
Traditional amino acid classifications are often based on physicochemical properties. This project takes a **data-driven** approach by using learned embeddings and VQ-VAE to discover latent groupings without supervision, potentially revealing novel insights that go beyond traditional classification schemes.

---

## 📦 Requirements
- Python 3.11

---

## 📁 Directory Structure
```bash
└── ./  
    ├── evaluation/                  # Evaluation scripts for clustering performance and visualization
    │   ├── cluster_eval.py          # Cluster evaluation metrics and analysis
    │   ├── dimension_reduction.py   # Dimensionality reduction techniques (e.g., PCA, t-SNE)
    │   └── plot.py                  # Plotting and visualization scripts for embedding spaces and cluster assignments
    ├── models/                      # VQ-VAE architecture and training scripts
    │   ├── __init__.py              # Initialization script for model package
    │   └── amino_clust.py           # Main VQ-VAE model for amino acid clustering
    ├── dataloader.py                # Data loading and preprocessing for amino acid sequences and embeddings
    ├── train.py                     # Script to train the VQ-VAE model
    └── utils.py                     # Utility functions for data processing, evaluation, and visualization
```
