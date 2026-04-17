# ================================
# 0. Import
# ================================
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# ================================
# 1. Generate Data
# ================================
np.random.seed(42)
N = 3000

X1 = np.random.normal(5, 2, N)
X2 = np.random.normal(3, 1.5, N)
X3 = np.random.normal(7, 2, N)
X4 = np.random.normal(4, 2, N)
X5 = np.random.normal(2, 1, N)

# Hidden true rule
y = (
    ((X1 > 6) & (X3 > 8)) |
    ((X2 < 2) & (X4 > 5)) |
    ((X1 > 7) & (X3 > 8) & (X5 > 1))
).astype(int)

# Add noise
flip = np.random.choice(N, int(0.1*N), replace=False)
y[flip] = 1 - y[flip]

X = pd.DataFrame({
    "X1": X1, "X2": X2, "X3": X3, "X4": X4, "X5": X5
})

# ================================
# 2. Train Deep Learning Model
# ================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train_s, y_train, epochs=20, verbose=0)

print("DL Test Accuracy:", model.evaluate(X_test_s, y_test, verbose=0)[1])

# ================================
# 3. Use DL as Teacher
# ================================
y_prob = model.predict(X_train_s).flatten()
y_teacher = (y_prob > 0.5).astype(int)

# ================================
# 4. Beam Search Rule Mining
# ================================
def beam_search_and_rules_df(X, y, beam_width=5, max_depth=3, min_support=0.05):
    
    features = X.columns.tolist()
    quantiles = np.linspace(0.2, 0.8, 7)
    th_grid = {f: np.quantile(X[f], quantiles) for f in features}
    
    beam = [([], np.ones(len(X), dtype=bool))]
    
    for depth in range(max_depth):
        candidates = []
        
        for rule, mask in beam:
            used = [r[0] for r in rule]
            
            for f in features:
                if f in used:
                    continue
                
                for th in th_grid[f]:
                    
                    new_rule = rule + [(f, int(np.round(th)))]
                    new_mask = mask & (X[f].values > th)
                    
                    if np.mean(new_mask) < min_support:
                        continue
                    
                    pred = new_mask.astype(int)
                    acc = np.mean(pred == y)
                    
                    candidates.append((new_rule, new_mask, acc))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = [(r, m) for r, m, s in candidates[:beam_width]]
    
    # =========================
    # 最終整理成 DataFrame
    # =========================
    rows = []
    
    for rule, mask in beam:
        pred = mask.astype(int)
        
        acc = np.mean(pred == y)
        precision = np.sum((pred==1)&(y==1)) / max(np.sum(pred==1),1)
        recall = np.sum((pred==1)&(y==1)) / max(np.sum(y==1),1)
        
        # ✅ F1 score
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        # rule string
        rule_str = " AND ".join([f"{f} > {th}" for f, th in rule])
        
        # 拆欄位（方便分析）
        row = {
            "rule": rule_str,
            "num_features": len(rule),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }
        
        # 每個 feature threshold 單獨欄位
        for f, th in rule:
            row[f] = th
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 排序（用 F1 比較合理）
    df = df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    
    return df

# ================================
# 5. Run Rule Mining
# ================================
rules = beam_search_and_rules(X_train, y_teacher, beam_width=5, max_depth=3)

print("\nTop Rules:\n")
for rule, acc, prec, rec, sup in rules[:5]:
    rule_str = " AND ".join([f"{f} > {th}" for f, th in rule])
    
    print(f"IF {rule_str} → Level A")
    print(f"Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Support: {sup:.4f}")
    print("-"*60)



rules_df = beam_search_and_rules_df(X_train, y_teacher, beam_width=5, max_depth=3)
rules_df
def beam_search_and_rules(X, y, beam_width=5, max_depth=3, min_support=0.05):
    
    features = X.columns.tolist()
    
    # threshold candidates (quantiles)
    quantiles = np.linspace(0.2, 0.8, 7)
    th_grid = {f: np.quantile(X[f], quantiles) for f in features}
    
    # initial beam
    beam = [([], np.ones(len(X), dtype=bool))]
    
    for depth in range(max_depth):
        candidates = []
        
        for rule, mask in beam:
            used = [r[0] for r in rule]
            
            for f in features:
                if f in used:
                    continue
                
                for th in th_grid[f]:
                    
                    new_rule = rule + [(f, int(np.round(th)))]
                    new_mask = mask & (X[f].values > th)
                    
                    # support filter
                    if np.mean(new_mask) < min_support:
                        continue
                    
                    pred = new_mask.astype(int)
                    score = np.mean(pred == y)
                    
                    candidates.append((new_rule, new_mask, score))
        
        # keep top-k
        candidates.sort(key=lambda x: x[2], reverse=True)
        beam = [(r, m) for r, m, s in candidates[:beam_width]]
    
    # evaluate
    results = []
    for rule, mask in beam:
        pred = mask.astype(int)
        
        acc = np.mean(pred == y)
        precision = np.sum((pred==1)&(y==1)) / max(np.sum(pred==1),1)
        recall = np.sum((pred==1)&(y==1)) / max(np.sum(y==1),1)
        support = np.mean(pred==1)
        
        results.append((rule, acc, precision, recall, support))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ================================
# 5. Run Rule Mining
# ================================
# rules = beam_search_and_rules(X_train, y_teacher, beam_width=5, max_depth=3)

print("\nTop Rules:\n")
for rule, acc, prec, rec, sup in rules[:5]:
    rule_str = " AND ".join([f"{f} > {th}" for f, th in rule])
    
    print(f"IF {rule_str} → Level A")
    print(f"Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | Support: {sup:.4f}")
    print("-"*60)

# ================================
# 6. Best Single Rule
# ================================
best_rule = rules[0][0]
rule_str = " AND ".join([f"{f} > {th}" for f, th in best_rule])

print("\n🔥 Final Rule:")
print(f"IF {rule_str} → Level A")

# ================================
# 7. Deployable Function
# ================================
def predict_rule(x):
    cond = True
    for f, th in best_rule:
        cond = cond and (x[f] > th)
    return int(cond)

# 測試
sample = X_test.iloc[0]
print("\nSample Prediction (Rule):", predict_rule(sample))
