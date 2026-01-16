import torch
import os
import sys
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoaderManager
from src.model import NeuMF
from src.modules.unigrf import UniGRF

def load_saved_model(model_path, num_users, num_items, embedding_dim, device):
    model = NeuMF(num_users, num_items, embedding_dim).to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    model.eval()
    return model

def main():
    # Configuration
    DATA_DIR = "d:/humou/recomend/ml-latest"
    MODEL_PATH = "d:/humou/recomend/model.pth" # Path to save/load model
    BATCH_SIZE = 1024
    EMBEDDING_DIM = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running on {DEVICE}")

    # 1. Data Loading (Need maps for ID conversion)
    # We need to load data to get the user/item mappings consistent with training
    dl_manager = DataLoaderManager(DATA_DIR, batch_size=BATCH_SIZE)
    dl_manager.load_data()
    
    # 2. Load Model
    model = load_saved_model(MODEL_PATH, dl_manager.num_users, dl_manager.num_items, EMBEDDING_DIM, DEVICE)

    # 3. Initialize UniGRF
    print("\nInitializing UniGRF for Recommendation...")
    all_item_embeddings = model.get_all_item_embeddings()
    unigrf = UniGRF(all_item_embeddings, embedding_dim=EMBEDDING_DIM)

    # 4. Interactive Loop
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

                # Recommend
                print(f"\n--- 为用户 {test_user_idx} 生成推荐 (UniGRF) ---")
                user_emb = model.get_user_embedding(test_user_idx)
                
                seen_items = set(user_items.tolist())
                candidates, _ = unigrf.retrieve(user_emb, k=100 + len(seen_items))
                filtered_candidates = [item for item in candidates if item not in seen_items]
                
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
                    
                print("\n正在生成新用户画像...")
                new_user_emb = model.get_new_user_embedding(new_user_items, new_user_ratings)
                
                print("正在检索推荐...")
                # Use GMF embedding for retrieval
                candidates, _ = unigrf.retrieve(new_user_emb['gmf'], k=100)
                
                rated_set = set(new_user_items)
                filtered_candidates = [item for item in candidates if item not in rated_set]
                
                # Use full embedding dict for ranking
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
