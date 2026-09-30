import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import ndcg_score
from data_utils import load_real_or_fallback_dataset, apply_position_based_click_model

def compute_dataset_ndcg(model, dataframe, feature_cols, k=5):
    """Calculates query-level average NDCG@k against true relevance labels."""
    dataframe = dataframe.copy()
    dataframe["pred_score"] = model.predict(dataframe[feature_cols])
    
    session_scores = []
    for _, group in dataframe.groupby("qid"):
        actual_relevance = group["true_rel"].to_numpy()
        predicted_outputs = group["pred_score"].to_numpy()
        
        if np.sum(actual_relevance) == 0:
            continue
            
        score = ndcg_score([actual_relevance], [predicted_outputs], k=k)
        session_scores.append(score)
        
    return np.mean(session_scores)

if __name__ == "__main__":
    print("==================================================================")
    print("Executing Version 2: Multi-Estimator Counterfactual Benchmark")
    print("==================================================================\n")
    
    if os.path.exists("mq2007_slice.txt"):
        os.remove("mq2007_slice.txt")
        
    raw_data = load_real_or_fallback_dataset("mq2007_slice.txt")
    biased_dataset = apply_position_based_click_model(raw_data)
    
    # 1. Propensity Weights
    biased_dataset["ips_weight"] = 1.0 / biased_dataset["propensity"]
    
    max_weight_cap = 3.0
    biased_dataset["clipped_ips_weight"] = np.minimum(biased_dataset["ips_weight"], max_weight_cap)
    
    biased_dataset = biased_dataset.sort_values("qid").reset_index(drop=True)
    
    # 2. Train/Test Split
    unique_qids = biased_dataset["qid"].unique()
    split_point = int(len(unique_qids) * 0.8)
    
    train_df = biased_dataset[biased_dataset["qid"].isin(unique_qids[:split_point])].copy()
    test_df = biased_dataset[biased_dataset["qid"].isin(unique_qids[split_point:])].copy()
    
    train_group_sizes = train_df.groupby("qid").size().to_numpy()
    feature_cols = [c for c in biased_dataset.columns if c.startswith("feat_")]
    
    # 3. Model Training
    print("1/3 Training Naive LambdaMART...")
    naive_ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", random_state=42, verbosity=-1)
    naive_ranker.fit(X=train_df[feature_cols], y=train_df["clicked"], group=train_group_sizes)
    
    print("2/3 Training Vanilla IPS LambdaMART...")
    ips_ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", random_state=42, verbosity=-1)
    ips_ranker.fit(X=train_df[feature_cols], y=train_df["clicked"], group=train_group_sizes, sample_weight=train_df["ips_weight"])
    
    print("3/3 Training Clipped IPS LambdaMART (M=3.0)...")
    clipped_ips_ranker = lgb.LGBMRanker(objective="lambdarank", metric="ndcg", random_state=42, verbosity=-1)
    clipped_ips_ranker.fit(X=train_df[feature_cols], y=train_df["clicked"], group=train_group_sizes, sample_weight=train_df["clipped_ips_weight"])
    
    # 4. Metric Evaluation
    naive_ndcg = compute_dataset_ndcg(naive_ranker, test_df, feature_cols, k=5)
    ips_ndcg = compute_dataset_ndcg(ips_ranker, test_df, feature_cols, k=5)
    clipped_ndcg = compute_dataset_ndcg(clipped_ips_ranker, test_df, feature_cols, k=5)
    
    print("\n======================= BENCHMARK RESULTS =======================")
    print(f"1. Baseline Naive LambdaMART NDCG@5:      {naive_ndcg:.4f}")
    print(f"2. Vanilla IPS LambdaMART NDCG@5:         {ips_ndcg:.4f}  (Lift: {((ips_ndcg - naive_ndcg)/naive_ndcg)*100:+.2f}%)")
    print(f"3. Clipped IPS LambdaMART NDCG@5:         {clipped_ndcg:.4f}  (Lift: {((clipped_ndcg - naive_ndcg)/naive_ndcg)*100:+.2f}%)")
    print("=================================================================\n")