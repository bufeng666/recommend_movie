from flask import Flask, render_template, request, jsonify
import torch
import sys
import os
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoaderManager
from src.model import NeuMF
from src.modules.unigrf import UniGRF

app = Flask(__name__)

# Global variables
dl_manager = None
model = None
unigrf = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "d:/humou/recomend/ml-latest"
MODEL_PATH = "d:/humou/recomend/model.pth"
EMBEDDING_DIM = 64

def init_system():
    global dl_manager, model, unigrf
    print("Initializing system...")
    
    # 1. Load Data
    dl_manager = DataLoaderManager(DATA_DIR, batch_size=1024)
    dl_manager.load_data()
    
    # 2. Load Model
    model = NeuMF(dl_manager.num_users, dl_manager.num_items, EMBEDDING_DIM).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("Warning: Model file not found. Please train the model first.")
    model.eval()
    
    # 3. Init UniGRF
    all_item_embeddings = model.get_all_item_embeddings()
    unigrf = UniGRF(all_item_embeddings, embedding_dim=EMBEDDING_DIM)
    print("System initialized.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search_movie')
def search_movie():
    query = request.args.get('query', '')
    if not query:
        return jsonify([])
    
    matches = dl_manager.movies_df[dl_manager.movies_df['title'].str.contains(query, case=False, na=False)]
    results = []
    for idx, row in matches.head(10).iterrows():
        results.append({
            'id': int(idx), # DataFrame index
            'title': row['title'],
            'genres': row['genres']
        })
    return jsonify(results)

@app.route('/recommend_existing', methods=['POST'])
def recommend_existing():
    try:
        user_id = int(request.form.get('user_id'))
        if user_id < 0 or user_id >= dl_manager.num_users:
            return jsonify({'error': 'User ID out of range'})
            
        # Get History
        user_mask = dl_manager.train_dataset.users == user_id
        user_items = dl_manager.train_dataset.items[user_mask]
        user_ratings = dl_manager.train_dataset.ratings[user_mask]
        
        history = []
        high_rated_indices = (user_ratings >= 4.0).nonzero(as_tuple=True)[0]
        for idx in high_rated_indices[:5]: # Top 5
            movie_id = user_items[idx].item()
            rating = user_ratings[idx].item()
            title = dl_manager.get_movie_title(movie_id)
            history.append({'title': title, 'rating': rating})
            
        # Recommend
        user_emb = model.get_user_embedding(user_id)
        seen_items = set(user_items.tolist())
        candidates, _ = unigrf.retrieve(user_emb, k=100 + len(seen_items))
        filtered_candidates = [item for item in candidates if item not in seen_items]
        
        ranked_items, ranked_scores = unigrf.rank(model, user_id, filtered_candidates, device=DEVICE)
        
        recommendations = []
        for i in range(min(10, len(ranked_items))):
            movie_idx = ranked_items[i]
            score = float(ranked_scores[i])
            title = dl_manager.get_movie_title(movie_idx)
            recommendations.append({'title': title, 'score': f"{score:.2f}"})
            
        return jsonify({
            'history': history,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/recommend_cold_start', methods=['POST'])
def recommend_cold_start():
    try:
        data = request.json
        ratings_data = data.get('ratings', [])
        
        if not ratings_data:
            return jsonify({'error': 'No ratings provided'})
            
        new_user_items = []
        new_user_ratings = []
        
        for item in ratings_data:
            df_idx = item['movie_idx']
            rating = float(item['rating'])
            
            # Convert DataFrame index to MovieID then to Internal Item Index
            original_movie_id = dl_manager.movies_df.loc[df_idx, 'movieId']
            if original_movie_id in dl_manager.item_map:
                internal_idx = dl_manager.item_map[original_movie_id]
                new_user_items.append(internal_idx)
                new_user_ratings.append(rating)
        
        if not new_user_items:
            return jsonify({'error': 'No valid movies found in training set'})

        # Generate Embedding
        new_user_emb = model.get_new_user_embedding(new_user_items, new_user_ratings)
        
        # Recommend
        # Use GMF embedding for retrieval (ANN)
        candidates, _ = unigrf.retrieve(new_user_emb['gmf'], k=100)
        rated_set = set(new_user_items)
        filtered_candidates = [item for item in candidates if item not in rated_set]
        
        # Use full embedding dict for ranking
        ranked_items, ranked_scores = unigrf.rank(model, new_user_emb, filtered_candidates, device=DEVICE)
        
        recommendations = []
        for i in range(min(10, len(ranked_items))):
            movie_idx = ranked_items[i]
            score = float(ranked_scores[i])
            title = dl_manager.get_movie_title(movie_idx)
            recommendations.append({'title': title, 'score': f"{score:.2f}"})
            
        return jsonify({'recommendations': recommendations})
        
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    init_system()
    app.run(debug=True, port=5000)
