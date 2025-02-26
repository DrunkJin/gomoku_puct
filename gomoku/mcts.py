import numpy as np
import math

class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.is_expanded = False
    
    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def expand(self, policy):
        """노드 확장"""
        legal_actions = self.state.get_legal_actions()
        policy_reshaped = policy.reshape(self.state.board_size, self.state.board_size)
        
        for action in legal_actions:
            if action not in self.children:
                next_state = self.state.copy()
                next_state.make_move(action)
                self.children[action] = Node(
                    next_state, 
                    parent=self,
                    action=action,
                    prior=policy_reshaped[action]
                )
        self.is_expanded = True

class PUCT:
    def __init__(self, network, num_simulations=800, c_puct=1.0):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, state):
        """루트 노드 반환을 위한 검색 함수"""
        root = Node(state)
        
        # MCTS 시뮬레이션 실행
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            # Selection
            while node.is_expanded:
                node = self._select_child(node)
                search_path.append(node)
            
            # Expansion and evaluation
            policy, value = self.network.predict(node.state)
            
            if not node.state.is_terminal():
                node.expand(policy)
            
            self._backpropagate(search_path, value)
            
        return root
    
    def get_action(self, state, temperature=1.0):
        root = self.search(state)
        return self._select_action(root, temperature)
    
    def _select_child(self, node):
        """PUCT 공식을 사용하여 자식 노드 선택"""
        best_score = float('-inf')
        best_action = None
        best_child = None
        
        for action, child in node.children.items():
            score = self._puct_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_child
    
    def _puct_score(self, parent, child):
        """PUCT 점수 계산"""
        prior_score = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)
        value_score = -child.value()  # 상대 플레이어 관점에서의 가치
        return value_score + prior_score
    
    def _backpropagate(self, search_path, value):
        """탐색 경로를 따라 가치 역전파"""
        for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            value = -value
    
    def _select_action(self, root, temperature):
        """방문 횟수를 기반으로 행동 선택"""
        visits = np.array([child.visit_count for child in root.children.values()])
        actions = list(root.children.keys())
        
        if temperature == 0:
            action_idx = np.argmax(visits)
            return actions[action_idx]
        
        # 온도를 적용한 확률 계산
        visits = visits ** (1 / temperature)
        probs = visits / np.sum(visits)
        action_idx = np.random.choice(len(actions), p=probs)
        
        return actions[action_idx]