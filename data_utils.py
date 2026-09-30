import numpy as np
import pandas as pd
import os

def parse_libsvm_line(line):
    parts = line.strip().split()
    if not parts:
        return None
    
    true_rel = int(parts[0])
    qid = parts[1].split(":")[1]
    
    features = {}
    for p in parts[2:]:
        if "#" in p:
            break
        fid, val = p.split(":")
        features[f"feat_{fid}"] = float(val)
        
    features["true_rel"] = true_rel
    features["qid"] = qid
    return features

def load_real_or_fallback_dataset(file_path="mq2007_slice.txt"):
    """Loads dataset or generates a LETOR structure with realistic noise levels."""
    if not os.path.exists(file_path):
        print(f"[{file_path} not found] Generating challenging 46-feature LETOR dataset...")
        np.random.seed(42)
        with open(file_path, "w") as f:
            for q in range(250):
                qid = f"100{q}"
                docs_per_query = np.random.randint(15, 30)
                for _ in range(docs_per_query):
                    rel = int(np.random.choice([0, 1, 2], p=[0.70, 0.20, 0.10]))
                    
                    feats = []
                    for i in range(1, 47):
                        # Blend true relevance signal with heavy noise to create realistic ranking difficulty
                        val = rel * np.random.uniform(0.1, 0.4) + np.random.normal(0, 1.0)
                        feats.append(f"{i}:{max(0.0, val):.4f}")
                        
                    feat_str = " ".join(feats)
                    f.write(f"{rel} qid:{qid} {feat_str} #docid=GX-mock\n")
                    
    data = []
    with open(file_path, "r") as f:
        for line in f:
            row = parse_libsvm_line(line)
            if row:
                data.append(row)
                
    return pd.DataFrame(data).fillna(0.0)

def apply_position_based_click_model(df):
    """Applies high-bias PBM click simulation."""
    np.random.seed(42)
    processed_sessions = []
    
    for qid, group in df.groupby("qid"):
        group = group.copy()
        
        # Severe presentation bias: legacy system ranks partially on noise
        noisy_ranking_score = np.random.normal(0, 2.0, size=len(group)) + group["true_rel"] * 0.5
        group = group.iloc[np.argsort(noisy_ranking_score)[::-1]].reset_index(drop=True)
        
        group["position"] = np.arange(len(group))
        
        # Strict examination decay curve
        group["propensity"] = 1.0 / np.power(group["position"] + 1, 0.8)
        
        # User intention probability
        intent_prob = 0.05 + 0.85 * (group["true_rel"] / 2.0)
        
        clicks = []
        for i in range(len(group)):
            p_click = group["propensity"].iloc[i] * intent_prob.iloc[i]
            clicks.append(1 if np.random.rand() < p_click else 0)
            
        group["clicked"] = clicks
        processed_sessions.append(group)
        
    return pd.concat(processed_sessions).reset_index(drop=True)