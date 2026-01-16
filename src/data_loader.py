import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

class MovieLensDataset(Dataset):
    def __init__(self, ratings_df):
        self.users = torch.tensor(ratings_df['user_idx'].values, dtype=torch.long)
        self.items = torch.tensor(ratings_df['movie_idx'].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

class DataLoaderManager:
    def __init__(self, data_dir, test_size=0.2, batch_size=1024):
        self.data_dir = data_dir
        self.test_size = test_size
        self.batch_size = batch_size
        self.num_users = 0
        self.num_items = 0
        self.user_map = {}
        self.item_map = {}
        self.movies_df = None
        
    def load_data(self):
        # Load Ratings
        print("Loading ratings...")
        ratings_path = f"{self.data_dir}/ratings.csv"
        # Reading only first 1M rows for faster development/testing if dataset is huge
        # Remove nrows=1000000 for full training
        ratings_df = pd.read_csv(ratings_path, nrows=1000000) 
        
        # Load Movies
        print("Loading movies...")
        movies_path = f"{self.data_dir}/movies.csv"
        self.movies_df = pd.read_csv(movies_path)
        
        # Map IDs to continuous indices
        print("Mapping IDs...")
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['movieId'].unique()
        
        self.user_map = {u: i for i, u in enumerate(unique_users)}
        self.item_map = {i: m for m, i in enumerate(unique_items)}
        
        # Reverse map for retrieval
        self.id_to_movie = {v: k for k, v in self.item_map.items()}
        
        self.num_users = len(unique_users)
        self.num_items = len(unique_items)
        
        ratings_df['user_idx'] = ratings_df['userId'].map(self.user_map)
        ratings_df['movie_idx'] = ratings_df['movieId'].map(self.item_map)
        
        # Split
        print("Splitting data...")
        train_df, test_df = train_test_split(ratings_df, test_size=self.test_size, random_state=42)
        
        self.train_dataset = MovieLensDataset(train_df)
        self.test_dataset = MovieLensDataset(test_df)
        
        print(f"Data loaded. Users: {self.num_users}, Items: {self.num_items}")
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def get_movie_title(self, movie_idx):
        if movie_idx not in self.id_to_movie:
            return "Unknown"
        original_id = self.id_to_movie[movie_idx]
        title = self.movies_df[self.movies_df['movieId'] == original_id]['title'].values
        return title[0] if len(title) > 0 else "Unknown"
