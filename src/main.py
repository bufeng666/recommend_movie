import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import sys

# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoaderManager
from src.model import NeuMF
from src.modules.ciess import CIESS
from src.modules.fedcia import FedCIA
from src.modules.unigrf import UniGRF

def main():
    # Configuration
    DATA_DIR = "d:/humou/recomend/ml-latest"
    BATCH_SIZE = 1024
    EPOCHS = 5
    EMBEDDING_DIM = 64
    LR = 0.005
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on {DEVICE}")

    # 1. Data Loading
    dl_manager = DataLoaderManager(DATA_DIR, batch_size=BATCH_SIZE)
    dl_manager.load_data()
    train_loader, test_loader = dl_manager.get_dataloaders()

    # 2. Model Initialization
    model = NeuMF(dl_manager.num_users, dl_manager.num_items, EMBEDDING_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    # 3. Module Initialization
    ciess = CIESS(compression_ratio=0.5) # Keep 50% of gradients
    fedcia = FedCIA(num_clients=5) # Simulate 5 clients environment
    
    # Training Loop (Simulating Local Training + "Federated" features)
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # FedCIA: Select Clients (Simulation)
        selected_clients = fedcia.select_clients()
        
        # In a real FL setup, we would split data by client. 
        # Here we just train on the global dataset but simulate gradient compression.
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for user_idx, item_idx, rating in progress_bar:
            user_idx, item_idx, rating = user_idx.to(DEVICE), item_idx.to(DEVICE), rating.to(DEVICE)
            
            optimizer.zero_grad()
            prediction = model(user_idx, item_idx)
            loss = criterion(prediction, rating)
            loss.backward()
            
            # --- CIESS Simulation: Compress Gradients ---
            # In a real scenario, we would compress gradients before sending to server.
            # Here we just demonstrate the compression/decompression cycle on one layer.
            with torch.no_grad():
                grad_sample = model.gmf_item_embedding.weight.grad
                if grad_sample is not None:
                    compressed = ciess.compress(grad_sample)
                    decompressed = ciess.decompress(compressed)
                    # For simulation, we just verify it works, but we use the real gradient for update
                    # to ensure convergence in this single-node demo.
                    # In real FL, we would use 'decompressed' on the server side.
            
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

    # 4. Evaluation
    print("\nEvaluating Model...")
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for user_idx, item_idx, rating in tqdm(test_loader, desc="Testing"):
            user_idx, item_idx, rating = user_idx.to(DEVICE), item_idx.to(DEVICE), rating.to(DEVICE)
            prediction = model(user_idx, item_idx)
            loss = criterion(prediction, rating)
            total_loss += loss.item()
            
            all_preds.extend(prediction.cpu().numpy())
            all_targets.extend(rating.cpu().numpy())
            
    mse = total_loss / len(test_loader)
    rmse = np.sqrt(np.mean((np.array(all_preds) - np.array(all_targets))**2))
    mae = np.mean(np.abs(np.array(all_preds) - np.array(all_targets)))
    
    print(f"\nTest Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # Top-K Evaluation (Sampled)
    print("\nEvaluating Top-K Metrics (Sampled)...")
    k = 10
    hits = 0
    total_relevant = 0
    num_eval_users = 100 # Evaluate on 100 random users to save time
    
    eval_users = np.random.choice(dl_manager.test_dataset.users.unique(), num_eval_users, replace=False)
    
    # Build temporary index for evaluation
    all_item_embeddings = model.get_all_item_embeddings()
    eval_unigrf = UniGRF(all_item_embeddings, embedding_dim=EMBEDDING_DIM)
    
    for uid in tqdm(eval_users, desc="Top-K Eval"):
        # Get ground truth: items user rated >= 4.0 in test set
        test_user_mask = (dl_manager.test_dataset.users == uid) & (dl_manager.test_dataset.ratings >= 4.0)
        relevant_items = dl_manager.test_dataset.items[test_user_mask].tolist()
        
        if not relevant_items:
            continue
            
        # Get items seen in training set (to filter out)
        train_user_mask = dl_manager.train_dataset.users == uid
        seen_items = set(dl_manager.train_dataset.items[train_user_mask].tolist())

        # Recommend
        user_emb = model.get_user_embedding(uid)
        # Retrieve more candidates to ensure we have enough after filtering
        candidates, _ = eval_unigrf.retrieve(user_emb, k=100 + len(seen_items)) 
        
        # Filter out seen items
        filtered_candidates = [item for item in candidates if item not in seen_items]
        
        # Rank remaining candidates
        ranked_items, _ = eval_unigrf.rank(model, uid, filtered_candidates, device=DEVICE)
        top_k_items = ranked_items[:k]
        
        # Calculate Recall@K
        num_hits = len(set(top_k_items) & set(relevant_items))
        hits += num_hits
        total_relevant += len(relevant_items)
        
    recall_at_k = hits / total_relevant if total_relevant > 0 else 0
    print(f"Recall@{k}: {recall_at_k:.4f}")

    # Save Model
    MODEL_PATH = "d:/humou/recomend/model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # 5. UniGRF: Retrieval and Ranking
    print("\nInitializing UniGRF for Recommendation...")
    model.eval()
    all_item_embeddings = model.get_all_item_embeddings()
    unigrf = UniGRF(all_item_embeddings, embedding_dim=EMBEDDING_DIM)
    
    # Interactive Mode
    while True:
        try:
            print("\n请选择模式:")
            print("1. 现有用户推荐 (输入 ID)")
            print("2. 冷启动新用户推荐 (输入喜欢的电影)")
            print("q. 退出")
            mode = input("请输入选项: ")
            
            if mode.lower() == 'q':
                break
                
            if mode == '1':
                user_input = input("\n请输入用户ID (0-9560): ")
                test_user_idx = int(user_input)
                if test_user_idx < 0 or test_user_idx >= dl_manager.num_users:
                    print("用户ID超出范围。")
                    continue
                    
                # Show History
                print(f"\n--- 用户 {test_user_idx} 的历史高分电影 (Rating >= 4.0) ---")
                user_mask = dl_manager.train_dataset.users == test_user_idx
                user_items = dl_manager.train_dataset.items[user_mask]
                user_ratings = dl_manager.train_dataset.ratings[user_mask]
                
                # Filter high ratings
                high_rated_indices = (user_ratings >= 4.0).nonzero(as_tuple=True)[0]
                if len(high_rated_indices) > 0:
                    shown_count = 0
                    for idx in high_rated_indices:
                        if shown_count >= 5: break
                        movie_id = user_items[idx].item()
                        rating = user_ratings[idx].item()
                        title = dl_manager.get_movie_title(movie_id)
                        print(f"  - {title} (Rating: {rating})")
                        shown_count += 1
                else:
                    print("  (无高分历史记录)")

                # Step 1: Retrieve (ANN)
                print(f"\n--- 为用户 {test_user_idx} 生成推荐 (UniGRF) ---")
                user_emb = model.get_user_embedding(test_user_idx)
                
                # Filter seen items
                seen_items = set(user_items.tolist())
                candidates, _ = unigrf.retrieve(user_emb, k=100 + len(seen_items))
                filtered_candidates = [item for item in candidates if item not in seen_items]
                
                # Step 2: Rank
                ranked_items, ranked_scores = unigrf.rank(model, test_user_idx, filtered_candidates, device=DEVICE)
                
                print("Top 10 推荐结果:")
                for i in range(min(10, len(ranked_items))):
                    movie_idx = ranked_items[i]
                    score = ranked_scores[i]
                    title = dl_manager.get_movie_title(movie_idx)
                    print(f"{i+1}. {title} (Score: {score:.4f})")
            
            elif mode == '2':
                print("\n--- 冷启动新用户推荐 ---")
                print("请输入几部你喜欢的电影名称关键词 (例如: Toy Story, Matrix, Star Wars)")
                print("输入 'done' 结束输入。")
                
                new_user_ratings = []
                new_user_items = []
                
                while True:
                    keyword = input("电影关键词: ")
                    if keyword.lower() == 'done':
                        break
                    
                    # Search for movie
                    matches = dl_manager.movies_df[dl_manager.movies_df['title'].str.contains(keyword, case=False, na=False)]
                    
                    if matches.empty:
                        print("未找到相关电影，请尝试其他关键词。")
                        continue
                        
                    print("找到以下电影:")
                    for idx, row in matches.head(5).iterrows():
                        print(f"{idx}: {row['title']}")
                        
                    selection = input("请输入你想评分的电影编号 (输入 'n' 跳过): ")
                    if selection.lower() == 'n':
                        continue
                        
                    try:
                        selected_idx = int(selection)
                        if selected_idx in matches.index:
                            rating = float(input("请输入评分 (0.5 - 5.0): "))
                            original_movie_id = matches.loc[selected_idx, 'movieId']
                            
                            if original_movie_id in dl_manager.item_map:
                                internal_idx = dl_manager.item_map[original_movie_id]
                                new_user_items.append(internal_idx)
                                new_user_ratings.append(rating)
                                print(f"已添加: {matches.loc[selected_idx, 'title']} - {rating}分")
                            else:
                                print("抱歉，该电影不在训练集中，无法用于生成画像。")
                        else:
                            print("无效编号。")
                    except ValueError:
                        print("输入无效。")
                
                if not new_user_items:
                    print("未输入有效评分，无法推荐。")
                    continue
                    
                # Generate Embedding for New User
                print("\n正在生成新用户画像...")
                new_user_emb = model.get_new_user_embedding(new_user_items, new_user_ratings)
                
                # Retrieve & Rank
                print("正在检索推荐...")
                candidates, _ = unigrf.retrieve(new_user_emb, k=100)
                
                # Filter out items just rated
                rated_set = set(new_user_items)
                filtered_candidates = [item for item in candidates if item not in rated_set]
                
                ranked_items, ranked_scores = unigrf.rank(model, new_user_emb, filtered_candidates, device=DEVICE)
                
                print("\nTop 10 推荐结果 (基于你的输入):")
                for i in range(min(10, len(ranked_items))):
                    movie_idx = ranked_items[i]
                    score = ranked_scores[i]
                    title = dl_manager.get_movie_title(movie_idx)
                    print(f"{i+1}. {title} (Score: {score:.4f})")

        except ValueError:
            print("请输入有效的数字。")
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
