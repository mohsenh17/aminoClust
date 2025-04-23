import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

def aa_property(dataframe):
    chargeDict = {'A':'Neutral','G':'Neutral','I':'Neutral','L':'Neutral','P':'Neutral',
                  'V':'Neutral','F':'Neutral','W':'Neutral','Y':'Neutral','C':'Neutral',
                  'M':'Neutral','S':'Neutral','T':'Neutral','R':'Positive','H':'Positive','K':'Positive',
                  'D':'Negative','E':'Negative','N':'Neutral','Q':'Neutral'}
    hydroIndexDict = {'F':100,'I':99,'W':97,'L':97,'V':76,'M':74,'Y':63,'C':49,'A':41,'T':13,'H':8,
                      'G':0,'S':-5,'Q':-10,'R':-14,'K':-23,'N':-28,'E':-31,'P':-46 ,'D':-55}
    massIndexDict = {'G':75, 'A':89, 'S':105, 'P':115, 'V':117, 'T':119, 'C':121, 'I':131, 
                     'L':131, 'N':132, 'D':133, 'B':133, 'K':146, 'Q':146, 'E':147, 'Z':147, 'M':149, 
                     'H':155, 'F':165, 'R':174, 'Y':181, 'W':204}
    
    propertyDict = { 'A': 'Aliphatic', 'G': 'Aliphatic', 'I': 'Aliphatic', 'L': 'Aliphatic', 
                    'P': 'Aliphatic', 'V': 'Aliphatic', 'F': 'Aromatic', 'W': 'Aromatic', 
                    'Y': 'Aromatic','C': 'Sulfur', 'M': 'Sulfur','S': 'Hydroxyl', 
                    'T': 'Hydroxyl','R': 'Basic', 'H': 'Basic', 'K': 'Basic','D': 'Acidic', 
                    'E': 'Acidic','N': 'Amide', 'Q': 'Amide'
                    }

    dataframe['charge'] = dataframe['label'].map(chargeDict)
    dataframe['hydrophobicity'] = dataframe['label'].map(hydroIndexDict)
    dataframe['mass'] = dataframe['label'].map(massIndexDict)
    dataframe['property'] = dataframe['label'].map(propertyDict)
    return dataframe
    

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

def plot_amino_acid_clusters(dataframe, cluster='unique', output_path="custom_markers_plot.png"):
    """
    Plots 2D embeddings of amino acids with custom markers and colors per cluster,
    and includes a legend for interpretation.

    Args:
        dataframe (pd.DataFrame): Must contain 'x', 'y', 'label', and cluster columns.
        cluster (str): Column name indicating cluster assignments.
        output_path (str): Path to save the output image.
    """

    # Define custom LaTeX-style markers for amino acids
    markers = {
        "A": "$A$", "C": "$C$", "D": "$D$", "E": "$E$", "F": "$F$",
        "G": "$G$", "H": "$H$", "I": "$I$", "K": "$K$", "L": "$L$",
        "M": "$M$", "N": "$N$", "P": "$P$", "Q": "$Q$", "R": "$R$",
        "S": "$S$", "T": "$T$", "V": "$V$", "W": "$W$", "Y": "$Y$",
    }

    # Assign a unique color to each cluster
    unique_clusters = sorted(dataframe[cluster].unique())
    cluster_to_color = {
        cid: cm.tab20(i / len(unique_clusters)) for i, cid in enumerate(unique_clusters)
    }

    plt.figure(figsize=(12, 10))

    # Plot each data point
    for _, row in dataframe.iterrows():
        x, y = row['x'], row['y']
        label = row['label']
        cluster_id = row[cluster]
        color = cluster_to_color[cluster_id]
        marker = markers.get(label, 'o')

        plt.scatter(x, y, color=color, marker=marker, s=200, edgecolor='black', linewidth=0.1)

        # Add to legend if not already included
        handles = [
            mpatches.Patch(color=color, label=str(cid)) 
            for cid, color in cluster_to_color.items()
        ]
        labels = [str(cid) for cid in cluster_to_color]

    # Add title and labels
    plt.title("Amino Acid Clusters with Custom Markers", fontsize=16)
    plt.xlabel("X-axis", fontsize=12)
    plt.ylabel("Y-axis", fontsize=12)
    plt.legend(handles=handles, labels=labels, title="Clusters")
    plt.grid(True)

    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

if __name__ == "__main__":
    import yaml
    import os
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    for j in [2, 4, 8, 16, 32, 64]:
        config['model']['latent_dim'] = j
        for i in range(2, 21):
            config['model']['num_clusters'] = i
            config['model']['name'] = f"aminoClust_{j}_{i}"
            
            tsne_path = config['base']['evaluation_dir'] + f"/tsne_latents_{config['model']['name']}.csv"
            if os.path.exists(tsne_path):
                print(f'tsne for {config['model']['name']} exist!')
                continue

            df = pd.read_csv(tsne_path)[:2000]
            df = aa_property(df)

            cluster_features = ['charge', 'hydrophobicity', 'mass', 'property', 'cluster_id']
            plot_subdir = config['base']['plot_dir'] + f"/{config['model']['name']}"
            os.makedirs(plot_subdir, exist_ok=True)

            for cluster in cluster_features:
                output_path = f"{plot_subdir}/{config['model']['name']}_{cluster}.png"
                plot_amino_acid_clusters(df, cluster=cluster, output_path=output_path)
    
    