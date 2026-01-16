import numpy as np
import torch

class FedCIA:
    """
    FedCIA: Federated Client Selection and Aggregation (Simplified Simulation)
    Focuses on Client Selection and Privacy-Preserving Aggregation
    """
    def __init__(self, num_clients=10, selection_ratio=0.5, privacy_noise_std=0.01):
        self.num_clients = num_clients
        self.selection_ratio = selection_ratio
        self.privacy_noise_std = privacy_noise_std
        self.clients = list(range(num_clients))
        self.client_scores = {i: 1.0 for i in range(num_clients)} # Dummy scores

    def select_clients(self):
        """
        Selects a subset of clients based on scores (e.g., data quality, resources).
        Here we simulate it by random selection weighted by scores.
        """
        num_selected = max(1, int(self.num_clients * self.selection_ratio))
        
        # Simple random selection for now
        selected_clients = np.random.choice(
            self.clients, 
            size=num_selected, 
            replace=False
        )
        print(f"[FedCIA] Selected clients: {selected_clients}")
        return selected_clients

    def secure_aggregate(self, model_updates):
        """
        Aggregates model updates with simulated Differential Privacy noise.
        model_updates: list of state_dicts
        """
        if not model_updates:
            return None
            
        aggregated_update = {}
        
        # Initialize with zeros
        first_update = model_updates[0]
        for key in first_update.keys():
            aggregated_update[key] = torch.zeros_like(first_update[key])
            
        # Sum updates
        for update in model_updates:
            for key in update.keys():
                aggregated_update[key] += update[key]
                
        # Average and Add Noise (Privacy)
        num_updates = len(model_updates)
        for key in aggregated_update.keys():
            # Average
            aggregated_update[key] /= num_updates
            
            # Add Gaussian Noise (Simulating DP)
            noise = torch.randn_like(aggregated_update[key]) * self.privacy_noise_std
            aggregated_update[key] += noise
            
        return aggregated_update
