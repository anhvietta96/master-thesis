import torch
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

# 1. Define amino acid order
aa_order = "ACDEFGHIKLMNPQRSTVWYXZ"
aa_to_idx = {aa: i for i, aa in enumerate(aa_order)}

# 2. Load BLOSUM62 (same as previous snippet)
blosum62_raw = [
    [ 4, 0,-2,-1,-2, 0,-2,-1,-1,-1,-1,-2,-1,-1,-1, 1, 0, 0,-3,-2,-1,-5],
    [ 0, 9,-3,-4,-2,-3,-3,-1,-3,-1,-1,-3,-3,-3,-3,-1,-1,-1,-2,-2,-3,-5],
    [-2,-3, 6, 2,-3,-1,-1,-3,-1,-4,-3, 1,-1, 0,-2, 0,-1,-3,-4,-3, 1,-5],
    [-1,-4, 2, 5,-3,-2, 0,-3, 1,-3,-2, 0,-1, 2, 0, 0,-1,-2,-3,-2, 4,-5],
    [-2,-2,-3,-3, 6,-3,-1, 0,-3, 0, 0,-3,-4,-3,-3,-2,-2,-1, 1, 3,-3,-5],
    [ 0,-3,-1,-2,-3, 6,-2,-4,-2,-4,-3, 0,-2,-2,-2, 0,-2,-3,-2,-3,-2,-5],
    [-2,-3,-1, 0,-1,-2, 8,-3,-1,-3,-2, 1,-2, 0, 0,-1,-2,-3,-2, 2, 0,-5],
    [-1,-1,-3,-3, 0,-4,-3, 4,-3, 2, 1,-3,-3,-3,-3,-2,-1, 3,-3,-1,-3,-5],
    [-1,-3,-1, 1,-3,-2,-1,-3, 5,-2,-1, 0,-1, 1, 2, 0,-1,-2,-3,-2, 1,-5],
    [-1,-1,-4,-3, 0,-4,-3, 2,-2, 4, 2,-3,-3,-2,-2,-2,-1, 1,-2,-1,-3,-5],
    [-1,-1,-3,-2, 0,-3,-2, 1,-1, 2, 5,-2,-2, 0,-1,-1,-1, 1,-1,-1,-1,-5],
    [-2,-3, 1, 0,-3, 0, 1,-3, 0,-3,-2, 6,-2, 0, 0, 1, 0,-3,-4,-2, 0,-5],
    [-1,-3,-1,-1,-4,-2,-2,-3,-1,-3,-2,-2, 7,-1,-2,-1,-1,-2,-4,-3,-1,-5],
    [-1,-3, 0, 2,-3,-2, 0,-3, 1,-2, 0, 0,-1, 5, 1, 0,-1,-2,-2,-1, 3,-5],
    [-1,-3,-2, 0,-3,-2, 0,-3, 2,-2,-1, 0,-2, 1, 5,-1,-1,-3,-3,-2, 0,-5],
    [ 1,-1, 0, 0,-2, 0,-1,-2, 0,-2,-1, 1,-1, 0,-1, 4, 1,-2,-3,-2, 0,-5],
    [ 0,-1,-1,-1,-2,-2,-2,-1,-1,-1,-1, 0,-1,-1,-1, 1, 5, 0,-2,-2,-1,-5],
    [ 0,-1,-3,-2,-1,-3,-3, 3,-2, 1, 1,-3,-2,-2,-3,-2, 0, 4,-3,-1,-2,-5],
    [-3,-2,-4,-3, 1,-2,-2,-3,-3,-2,-1,-4,-4,-2,-3,-3,-2,-3,11, 2,-3,-5],
    [-2,-2,-3,-2, 3,-3, 2,-1,-2,-1,-1,-2,-3,-1,-2,-2,-2,-1, 2, 7,-2,-5],
    [-1,-3, 1, 4,-3,-2, 0,-3, 1,-3,-1, 0,-1, 3, 0, 0,-1,-2,-3,-2, 4,-5],
    [-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,-5,12],  # Z (used for X)
]

blosum = torch.tensor(blosum62_raw, dtype=torch.float)

# Normalize distances
distance_matrix = (blosum - blosum.min()) / (blosum.max() - blosum.min())
# Convert similarity to distance: higher score → smaller distance
distance_matrix = -blosum

print(distance_matrix)
'''
# 3. Train SOM
input_dim = 5  # dimension of latent embedding
som = MiniSom(x=5, y=5, input_len=input_dim, sigma=7.1, learning_rate=0.5, random_seed=42)

# Initialize with PCA for stability
pca = PCA(n_components=input_dim)
init_weights = pca.fit_transform(distance_matrix.numpy())
som.weights = init_weights.reshape(5, 5, input_dim)

# Train SOM on BLOSUM-derived distances
som.train(data=distance_matrix.numpy(), num_iteration=1000, verbose=True)

# 4. Map amino acids to their embeddings
embeddings = {}
for i, aa in enumerate(aa_order):
    winner = som.winner(distance_matrix[i].numpy())
    embedding = som.get_weights()[winner[0], winner[1]]
    embeddings[aa] = embedding

# Optional: visualize with PCA
emb_matrix = np.array([embeddings[aa] for aa in aa_order])
emb_2d = PCA(n_components=2).fit_transform(emb_matrix)

plt.figure(figsize=(8, 6))
for i, aa in enumerate(aa_order):
    plt.scatter(emb_2d[i, 0], emb_2d[i, 1], label=aa)
    plt.text(emb_2d[i, 0], emb_2d[i, 1], aa)
plt.title("SOM-based amino acid embeddings (2D PCA projection)")
plt.grid()
plt.show()
'''

embedding_dim = 3
close = False
while embedding_dim < 6:   # target embedding space
    mds = MDS(n_components=embedding_dim, dissimilarity='precomputed', random_state=42)
    mds_embeddings = mds.fit_transform(distance_matrix)

    # Result: mds_embeddings[i] is the 5D embedding of aa_order[i]
    embeddings = {aa: mds_embeddings[i] for i, aa in enumerate(aa_order)}

    pca = PCA(n_components=2)
    mds_2d = pca.fit_transform(mds_embeddings)

    plt.figure(figsize=(8,6))
    for i, aa in enumerate(aa_order):
        plt.scatter(mds_2d[i, 0], mds_2d[i, 1])
        plt.text(mds_2d[i, 0], mds_2d[i, 1], aa)
    plt.title("2D Projection of MDS Embeddings")
    plt.grid(True)

    close = np.allclose()
    plt.show()

    embedding_dim += 1

