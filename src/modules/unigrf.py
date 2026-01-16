import faiss
import torch
import numpy as np

class UniGRF:
    """
    UniGRF: Unified Generative Retrieval Framework (Simplified)
    Focuses on Retrieval (ANN) and Ranking.
    """
    def __init__(self, item_embeddings, embedding_dim=64):
        """
        item_embeddings: Tensor of shape (num_items, embedding_dim)
        """
        self.embedding_dim = embedding_dim
        self.index = None
        self.update_index(item_embeddings)

    def update_index(self, item_embeddings):
        """
        Builds/Updates the FAISS index for retrieval.
        """
        # FAISS expects numpy float32
        if torch.is_tensor(item_embeddings):
            embeddings_np = item_embeddings.cpu().detach().numpy().astype('float32')
        else:
            embeddings_np = item_embeddings.astype('float32')
            
        # L2 Distance Index (Euclidean) - equivalent to Max Inner Product if normalized
        # For simple MF dot product, Inner Product index is better
        self.index = faiss.IndexFlatIP(self.embedding_dim) 
        self.index.add(embeddings_np)
        print(f"[UniGRF] Index updated with {embeddings_np.shape[0]} items.")

    def retrieve(self, user_embedding, k=50):
        """
        Retrieves top-k candidate items for a user.
        """
        if torch.is_tensor(user_embedding):
            user_emb_np = user_embedding.cpu().detach().numpy().astype('float32')
        else:
            user_emb_np = user_embedding.astype('float32')
            
        # Reshape if single vector
        if len(user_emb_np.shape) == 1:
            user_emb_np = user_emb_np.reshape(1, -1)
            
        distances, indices = self.index.search(user_emb_np, k)
        return indices[0], distances[0]

    def rank(self, model, user_input, candidate_indices, device='cpu'):
        """
        Re-ranks the retrieved candidates using the full model.
        user_input: Can be an integer (user_idx) or a tensor (user_embedding) for new users.
        """
        item_tensor = torch.tensor(candidate_indices).to(device)
        
        with torch.no_grad():
            if isinstance(user_input, (int, np.integer)):
                # Existing user: use model's forward pass which looks up embedding
                user_tensor = torch.tensor([user_input] * len(candidate_indices)).to(device)
                scores = model(user_tensor, item_tensor)
            else:
                # New user: user_input is a dict {'gmf': ..., 'mlp': ...}
                gmf_user = user_input['gmf'].to(device)
                mlp_user = user_input['mlp'].to(device)
                
                # Expand to match candidates
                gmf_user = gmf_user.expand(len(candidate_indices), -1)
                mlp_user = mlp_user.expand(len(candidate_indices), -1)
                
                # Get Item Embeddings
                gmf_items = model.gmf_item_embedding(item_tensor)
                mlp_items = model.mlp_item_embedding(item_tensor)
                
                # Forward pass logic manually
                # GMF
                gmf_vector = gmf_user * gmf_items
                
                # MLP
                mlp_vector = torch.cat([mlp_user, mlp_items], dim=1)
                mlp_vector = model.mlp_layers(mlp_vector)
                
                # Concat
                vector = torch.cat([gmf_vector, mlp_vector], dim=1)
                
                # Predict
                scores = model.predict_layer(vector).squeeze()
            
        # Sort by score descending
        sorted_indices = torch.argsort(scores, descending=True)
        ranked_candidates = item_tensor[sorted_indices].cpu().numpy()
        ranked_scores = scores[sorted_indices].cpu().numpy()
        
        return ranked_candidates, ranked_scores
