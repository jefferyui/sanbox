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

# =========================================
# 1. Generate Data
# =========================================
N = 3000

X = pd.DataFrame({
    "X1": np.random.normal(5,2,N),
    "X2": np.random.normal(3,1.5,N),
    "X3": np.random.normal(7,2,N),
    "X4": np.random.normal(4,2,N),
    "X5": np.random.normal(2,1,N),
})

# hidden rule (含 interaction + noise)
y = (
    ((X["X1"]>6)&(X["X3"]>8)) |
    ((X["X2"]<2)&(X["X4"]>5)) |
    ((X["X1"]>7)&(X["X3"]>8)&(X["X5"]>1))
).astype(int)

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

# teacher label（去 noise）
y_teacher = (model.predict(X_train_s).flatten() > 0.5).astype(int)

# =========================================
# 3. Utils
# =========================================
def apply_rule(X, rule):
    mask = np.ones(len(X), dtype=bool)
    for f, th in rule:
        mask &= (X[f].values > th)
    return mask

def compute_metrics(mask, y):
    pred = mask.astype(int)
    acc = np.mean(pred == y)
    precision = np.sum((pred==1)&(y==1)) / max(np.sum(pred==1),1)
    recall = np.sum((pred==1)&(y==1)) / max(np.sum(y==1),1)
    f1 = 0 if precision+recall==0 else 2*precision*recall/(precision+recall)
    return acc, precision, recall, f1

def rule_to_str(rule):
    return " AND ".join([f"{f}>{th}" for f,th in rule])

# =========================================
# 4. Beam Search (AND rule)
# =========================================
def beam_search_df(X, y, beam_width=5, max_depth=3):
    
    qs = np.linspace(0.2,0.8,5)
    grid = {f: np.quantile(X[f], qs) for f in X.columns}
    
    beam = [([], np.ones(len(X), dtype=bool))]
    
    for _ in range(max_depth):
        candidates = []
        
        for rule, mask in beam:
            used = [r[0] for r in rule]
            
            for f in X.columns:
                if f in used:
                    continue
                
                for th in grid[f]:
                    new_rule = rule + [(f, int(round(th)))]
                    new_mask = mask & (X[f].values > th)
                    
                    acc,_,_,_ = compute_metrics(new_mask, y)
                    candidates.append((new_rule, new_mask, acc))
        
        candidates.sort(key=lambda x:x[2], reverse=True)
        beam = [(r,m) for r,m,s in candidates[:beam_width]]
    
    rows = []
    for r,m in beam:
        acc,p,rcl,f1 = compute_metrics(m,y)
        rows.append({
            "method":"Beam",
            "rule":rule_to_str(r),
            "num_features":len(r),
            "accuracy":acc,
            "precision":p,
            "recall":rcl,
            "f1":f1
        })
    
    return pd.DataFrame(rows).sort_values("f1", ascending=False)

# =========================================
# 5. MCTS (正確 AND rule)
# =========================================
class Node:
    def __init__(self, rule, parent=None):
        self.rule = rule
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def ucb(node, c=1.4):
    if node.visits == 0:
        return float("inf")
    return node.value/node.visits + c*np.sqrt(np.log(node.parent.visits+1)/node.visits)

def select(node):
    while node.children:
        node = max(node.children, key=lambda n: ucb(n))
    return node

def expand(node, X, grid):
    used = [r[0] for r in node.rule]
    for f in X.columns:
        if f in used:
            continue
        for th in grid[f]:
            node.children.append(Node(node.rule + [(f,int(round(th)))], node))

def backprop(node, reward):
    while node:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts_df(X, y, iters=500, max_depth=3):
    
    qs = np.linspace(0.2,0.8,5)
    grid = {f: np.quantile(X[f], qs) for f in X.columns}
    
    root = Node([])
    
    best_node = None
    best_score = -1
    
    for _ in range(iters):
        
        node = select(root)
        
        if len(node.rule) < max_depth:
            expand(node, X, grid)
            if node.children:
                node = random.choice(node.children)
        
        mask = apply_rule(X, node.rule)
        _,_,_,f1 = compute_metrics(mask, y)
        
        if f1 > best_score:
            best_score = f1
            best_node = node
        
        backprop(node, f1)
    
    # collect best few nodes（從 tree 抽）
    results = []
    
    def traverse(n):
        if n.rule:
            mask = apply_rule(X, n.rule)
            acc,p,rcl,f1 = compute_metrics(mask,y)
            results.append({
                "method":"MCTS",
                "rule":rule_to_str(n.rule),
                "num_features":len(n.rule),
                "accuracy":acc,
                "precision":p,
                "recall":rcl,
                "f1":f1
            })
        for c in n.children:
            traverse(c)
    
    traverse(root)
    
    df = pd.DataFrame(results).drop_duplicates("rule")
    return df.sort_values("f1", ascending=False)

# =========================================
# 6. Run Comparison
# =========================================
df_beam = beam_search_df(X_train, y_teacher)
df_mcts = mcts_df(X_train, y_teacher)

df_all = pd.concat([df_beam, df_mcts]).sort_values("f1", ascending=False)

print("\n===== Top Rules =====\n")
print(df_all.head(10))
