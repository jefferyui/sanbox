import math
import random

# ==========================================
# 1. 基礎環境與評分函數設定
# ==========================================

# 所有可能的候選條件 (Search Space)
CANDIDATES = ['X1>6', 'X2>5', 'X3>8', 'X4>2', 'X5>1']
MAX_DEPTH = 3 # 我們規定 AND rule 最多由 3 個條件組成

def evaluate_rule(rule_list):
    """
    模擬計算 AND rule 的最終得分。
    使用 set 來忽略條件的順序 (X1 AND X2 等同於 X2 AND X1)
    """
    rule_set = frozenset(rule_list)
    
    # 🚨 陷阱路徑 (初期分數高，誘使貪婪演算法選擇)
    if rule_set == frozenset(['X2>5']): return 0.6
    if rule_set == frozenset(['X2>5', 'X4>2']): return 0.8
    if rule_set == frozenset(['X2>5', 'X4>2', 'X5>1']): return 0.85 # 陷阱的極限
    
    # 🏆 真實目標 (初期分數低，但組合起來是全局最優)
    if rule_set == frozenset(['X1>6']): return 0.3
    if rule_set == frozenset(['X1>6', 'X3>8']): return 0.5
    if rule_set == frozenset(['X1>6', 'X3>8', 'X5>1']): return 1.0 # 真正的寶藏
    
    # 其他沒有意義的組合給予基礎低分
    return 0.1 * len(rule_list)


# ==========================================
# 2. Beam Search (束搜索) 實作
# ==========================================

def beam_search(beam_width=1):
    # 初始狀態是空的 rule，分數為 0
    beam = [([], 0.0)]  
    
    for depth in range(MAX_DEPTH):
        next_beam = []
        for current_rule, _ in beam:
            # 展開所有尚未加入的候選條件
            for cand in CANDIDATES:
                if cand not in current_rule:
                    new_rule = sorted(current_rule + [cand]) # 排序避免重複組合
                    
                    # 確保沒有重複放入 next_beam
                    if new_rule not in [item[0] for item in next_beam]:
                        score = evaluate_rule(new_rule)
                        next_beam.append((new_rule, score))
        
        # 依照分數由高到低排序，並「剪枝」只保留前 beam_width 名
        next_beam.sort(key=lambda x: x[1], reverse=True)
        beam = next_beam[:beam_width]
        
    return beam[0] # 回傳最高分的那一組


# ==========================================
# 3. MCTS (蒙地卡羅樹搜索) 實作
# ==========================================

class MCTSNode:
    def __init__(self, state, parent=None):
        self.state = state         # 當前的規則組合 (tuple)
        self.parent = parent
        self.children = {}         # key: 加入的條件, value: MCTSNode
        self.visits = 0            # 訪問次數 (N)
        self.value = 0.0           # 總回報分數 (Q)
        self.untried_moves = [c for c in CANDIDATES if c not in state]
        
    def ucb1(self, c=1.414):
        """UCT 公式計算"""
        if self.visits == 0:
            return float('inf')
        return (self.value / self.visits) + c * math.sqrt(math.log(self.parent.visits) / self.visits)

def mcts_search(iterations=500):
    root = MCTSNode(tuple())
    
    for _ in range(iterations):
        node = root
        
        # 1. Select (選擇)
        while not node.untried_moves and node.children and len(node.state) < MAX_DEPTH:
            node = max(node.children.values(), key=lambda n: n.ucb1())
            
        # 2. Expand (展開)
        if node.untried_moves and len(node.state) < MAX_DEPTH:
            move = random.choice(node.untried_moves)
            node.untried_moves.remove(move)
            new_state = tuple(sorted(list(node.state) + [move]))
            child = MCTSNode(new_state, parent=node)
            node.children[move] = child
            node = child
            
        # 3. Rollout (模擬)
        current_state = list(node.state)
        available_moves = [c for c in CANDIDATES if c not in current_state]
        while len(current_state) < MAX_DEPTH and available_moves:
            move = random.choice(available_moves)
            current_state.append(move)
            available_moves.remove(move)
            
        score = evaluate_rule(current_state)
        
        # 4. Backpropagate (回傳更新)
        while node is not None:
            node.visits += 1
            node.value += score
            node = node.parent
            
    # 根據訪問次數 (visits) 決定最終的最佳路徑
    best_rule = []
    current_node = root
    while current_node.children:
        best_move = max(current_node.children.items(), key=lambda item: item[1].visits)[0]
        best_rule.append(best_move)
        current_node = current_node.children[best_move]
        
    return best_rule, evaluate_rule(best_rule)


# ==========================================
# 4. 執行與結果輸出
# ==========================================

if __name__ == "__main__":
    def format_rule(rule_list):
        return " AND ".join(rule_list)

    print("=== Beam Search 測試 ===")
    
    # 測試 1：純貪婪 (k=1)
    rule_k1, score_k1 = beam_search(beam_width=1)
    print(f"[Beam Width=1] 找到規則: {format_rule(rule_k1)} (分數: {score_k1})")
    
    # 測試 2：足夠的搜尋寬度 (k=3)
    rule_k3, score_k3 = beam_search(beam_width=5)
    print(f"[Beam Width=3] 找到規則: {format_rule(rule_k3)} (分數: {score_k3})")
    
    print("\n=== MCTS 測試 ===")
    
    # MCTS 測試 (執行 200 次迭代)
    rule_mcts, score_mcts = mcts_search(iterations=200)
    print(f"[MCTS (200次迭代)] 找到規則: {format_rule(rule_mcts)} (分數: {score_mcts})")
