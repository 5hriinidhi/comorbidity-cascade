import sys
import yaml
import torch
import pandas as pd
import sklearn
import catboost
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score

def main():
    try:
        print("--- SMOKE TEST ---")
        print(f"PyTorch Version: {torch.__version__}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device availability: {device.upper()}")
        
        tensor_a = torch.randn(256, 128, device=device)
        tensor_b = torch.randn(128, 64, device=device)
        result = torch.matmul(tensor_a, tensor_b)
        print(f"Tensor Matmul Shape: {result.shape}")
        
        print("Training CatBoostClassifier on 100 random samples...")
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, size=100)
        
        model = CatBoostClassifier(iterations=10, logging_level='Silent')
        model.fit(X, y)
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        
        print(f"CatBoost Accuracy (Train): {acc:.2f}")
        
        print("ALL OK")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
