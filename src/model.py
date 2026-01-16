import torch
import torch.nn as nn

class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, layers=[64, 32, 16]):
        super(NeuMF, self).__init__()
        
        # GMF Part (Generalized Matrix Factorization)
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP Part (Multi-Layer Perceptron)
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        mlp_modules = []
        input_size = embedding_dim * 2
        for output_size in layers:
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(0.2))
            input_size = output_size
        self.mlp_layers = nn.Sequential(*mlp_modules)
        
        # Final Prediction Layer
        # Input: GMF output (dim) + MLP output (last layer size)
        self.predict_layer = nn.Linear(embedding_dim + layers[-1], 1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
        
        for m in self.mlp_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def forward(self, user_indices, item_indices):
        # GMF
        gmf_u = self.gmf_user_embedding(user_indices)
        gmf_i = self.gmf_item_embedding(item_indices)
        gmf_vector = gmf_u * gmf_i
        
        # MLP
        mlp_u = self.mlp_user_embedding(user_indices)
        mlp_i = self.mlp_item_embedding(item_indices)
        mlp_vector = torch.cat([mlp_u, mlp_i], dim=1)
        mlp_vector = self.mlp_layers(mlp_vector)
        
        # Concatenate
        vector = torch.cat([gmf_vector, mlp_vector], dim=1)
        
        # Predict
        prediction = self.predict_layer(vector)
        return prediction.squeeze()

    def get_user_embedding(self, user_idx):
        # For retrieval, we can use the GMF embedding as a proxy for "preference vector"
        # Or concatenate both. For simplicity and compatibility with dot-product ANN,
        # we'll use GMF embedding primarily, or we need a more complex retrieval strategy.
        # Here we return GMF embedding for ANN search.
        device = self.gmf_user_embedding.weight.device
        return self.gmf_user_embedding(torch.tensor([user_idx], device=device))

    def get_new_user_embedding(self, rated_item_indices, ratings):
        device = self.gmf_item_embedding.weight.device
        item_indices = torch.tensor(rated_item_indices, device=device)
        ratings_tensor = torch.tensor(ratings, device=device, dtype=torch.float32).unsqueeze(1)
        
        total_weight = ratings_tensor.sum()
        
        # 1. GMF Approximation
        gmf_item_embeds = self.gmf_item_embedding(item_indices)
        if total_weight > 0:
            gmf_user_emb = (gmf_item_embeds * ratings_tensor).sum(dim=0) / total_weight
        else:
            gmf_user_emb = self.gmf_item_embedding.weight.mean(dim=0)

        # 2. MLP Approximation
        mlp_item_embeds = self.mlp_item_embedding(item_indices)
        if total_weight > 0:
            mlp_user_emb = (mlp_item_embeds * ratings_tensor).sum(dim=0) / total_weight
        else:
            mlp_user_emb = self.mlp_item_embedding.weight.mean(dim=0)
            
        return {
            'gmf': gmf_user_emb.unsqueeze(0),
            'mlp': mlp_user_emb.unsqueeze(0)
        }

    def get_all_item_embeddings(self):
        # Similarly, return GMF item embeddings for ANN index
        return self.gmf_item_embedding.weight.data
