import faiss
import numpy as np


class FaissNN:
    def __init__(self, embedding_dim, use_gpu=False, metric="L2"):

        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu

        # Choose the appropriate index based on the metric
        if metric == "L2":
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (Euclidean)
        elif metric == "IP":
            self.index = faiss.IndexFlatIP(
                embedding_dim
            )  # Inner Product (Cosine Similarity when embeddings are normalized)
        else:
            raise ValueError("Metric must be 'L2' or 'IP'")

        # If GPU is enabled, move the index to GPU
        if self.use_gpu:
            res = faiss.StandardGpuResources()  # Initialize GPU resources
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # 0 is the GPU ID

    def add_embeddings(self, embeddings):

        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)  # Add the embeddings to the FAISS index

    def search(self, query_embedding, k=5):

        query_embedding = (
            np.array(query_embedding).reshape(1, -1).astype("float32")
        )  # Ensure correct shape and dtype
        distances, indices = self.index.search(query_embedding, k)  # Perform the search
        return distances, indices
