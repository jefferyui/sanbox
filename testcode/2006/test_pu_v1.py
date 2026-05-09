# =========================================
# 0. Import
# =========================================
import numpy as np
import pandas as pd
import tensorflow as tf
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# =========================================
# 1. Data
# =========================================
N = 3000
X = pd.DataFrame({
    "X1": np.random.normal(5,2,N),
    "X2": np.random.normal(3,1.5,N),
    "X3": np.random.normal(7,2,N),
    "X4": np.random.normal(4,2,N),
    "X5": np.random.normal(2,1,N),
})

# hidden rules（含 interaction）
y = (
    ((X["X1"]>6)&(X["X3"]>8)) |
    ((X["X2"]<2)&(X["X4"]>5)) |
    ((X["X1"]>7)&(X["X3"]>8)&(X["X5"]>1))
).astype(int)

# noise
flip = np.random.choice(N, int(0.1*N), replace=False)
y.iloc[flip] = 1 - y.iloc[flip]

# =========================================
# 2. DL Teacher
# =========================================
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
model.fit(X_train_s, y_train, epochs=15, verbose=0)

print("DL Test Acc:", model.evaluate(X_test_s, y_test, verbose=0)[1])

# teacher labels（去噪）
y_teacher = (model.predict(X_train_s).flatten() > 0.5).astype(int)

# =========================================
# 3. Prior（Gradient × Input 近似 SHAP）
# =========================================
def compute_prior(model, X_scaled):
    X_tensor = tf.convert_to_tensor(X_scaled, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        preds = model(X_tensor)
    grads = tape.gradient(preds, X_tensor).numpy()
    
    importance = np.mean(np.abs(grads * X_scaled), axis=0)
    importance = importance / (importance.sum() + 1e-8)
    
    return importance

feature_names = X.columns.tolist()
prior_vec = compute_prior(model, X_train_s)
prior_dict = dict(zip(feature_names, prior_vec))

print("Feature Prior:", prior_dict)

# =========================================
# 4. Utils
# =========================================
def apply_rule(X, rule):
    mask = np.ones(len(X), dtype=bool)
    for f, th in rule:
        mask &= (X[f].values > th)
    return mask

def compute_metrics(mask, y):
    pred = mask.astype(int)
    precision = np.sum((pred==1)&(y==1)) / max(np.sum(pred==1),1)
    recall = np.sum((pred==1)&(y==1)) / max(np.sum(y==1),1)
    f1 = 0 if precision+recall==0 else 2*precision*recall/(precision+recall)
    return precision, recall, f1

def rule_to_str(rule):
    return " AND ".join([f"{f}>{round(th,2)}" for f,th in rule])

# =========================================
# 5. Adaptive Grid（初始 + 可擴充）
# =========================================
class AdaptiveGrid:
    def __init__(self, X, bins=5):
        self.X = X
        self.grid = {}
        qs = np.linspace(0.2, 0.8, bins)
        for f in X.columns:
            self.grid[f] = list(np.quantile(X[f], qs))
    
    def get(self, f):
        return self.grid[f]
    
    def refine(self, f, center, radius=0.5, n=5):
        pts = np.linspace(center-radius, center+radius, n)
        self.grid[f].extend(pts)
        self.grid[f] = list(np.unique(np.round(self.grid[f], 3)))

# =========================================
# 6. PUCT MCTS
# =========================================
class Node:
    def __init__(self, rule, parent=None):
        self.rule = rule
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def puct_score(node, prior, c=1.5):
    if node.visits == 0:
        return float("inf")
    
    f = node.rule[-1][0] if node.rule else None
    p = prior.get(f, 1e-3)
    
    return (node.value/node.visits) + c * p * np.sqrt(node.parent.visits+1)/(1+node.visits)

def select(node, prior):
    while node.children:
        node = max(node.children, key=lambda n: puct_score(n, prior))
    return node

def expand(node, X, grid):
    used = [r[0] for r in node.rule]
    for f in X.columns:
        if f in used:
            continue
        for th in grid.get(f):
            node.children.append(Node(node.rule+[(f,th)], node))

def backprop(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

# =========================================
# 7. MCTS + PUCT + Adaptive Grid
# =========================================
def mcts_puct(X, y, prior, iters=400, max_depth=3):
    
    grid = AdaptiveGrid(X)
    root = Node([])
    
    best_rule = None
    best_f1 = -1
    
    for _ in range(iters):
        
        # Selection
        node = select(root, prior)
        
        # Expansion
        if len(node.rule) < max_depth:
            expand(node, X, grid)
            if node.children:
                node = random.choice(node.children)
        
        # Evaluate
        mask = apply_rule(X, node.rule)
        _,_,f1 = compute_metrics(mask, y)
        
        # Adaptive refine（只在好 rule 附近）
        if f1 > 0.7:
            for f,th in node.rule:
                grid.refine(f, th, radius=0.3)
        
        # Record best
        if f1 > best_f1:
            best_f1 = f1
            best_rule = node.rule
        
        # Backprop
        backprop(node, f1)
    
    return best_rule, best_f1

# =========================================
# 8. Run
# =========================================
rule, f1 = mcts_puct(X_train, y_teacher, prior_dict, iters=500)

print("\n=== Best Rule (PUCT) ===")
print(rule_to_str(rule))
print("F1:", f1)
