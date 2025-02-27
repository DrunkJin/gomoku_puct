import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import os
from torch.utils.data import Dataset, DataLoader

class ReplayBuffer:
    def __init__(self, max_size = 100_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, policy, value):
        self.buffer.append((state, policy, value))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class GomokuDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        return (self.states[idx], self.policies[idx], self.values[idx])
    
class Trainer:
    def __init__(self, network, device='cuda', lr=0.001, weight_decay=1e-4):
        self.network = network
        self.device = device
        self.optimizer = torch.optim.Adam(
            self.network.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        self.buffer = ReplayBuffer()

    def generate_self_play_data(self, mcts, num_games=100, temperature_cutoff=10):
        # 자가 대국을 통한 훈련 데이터 생성
        from gomoku_env import GomokuState

        for game_idx in range(num_games):
            print(f"Generating self-play games {game_idx + 1}/{num_games}")
            state = GomokuState()
            game_states, game_policies, game_values = [], [], []

            # 게임 플레이
            while not state.is_terminal():
                # 현재 상태 저장
                game_states.append(state.get_encoded_state())

                # MCTS로 정책 생성
                temperature = 1.0 if state.move_count < temperature_cutoff else 0.1
                action = mcts.get_action(state, temperature)

                # 정책 저장 (방문 횟수 기반)
                policy = np.zeros(state.board_size * state.board_size)
                root = mcts.search(state)

                for child_action, child_node in root.children.items():
                    idx = child_action[0] * state.board_size + child_action[1]
                    policy[idx] = child_node.visit_count

                # 정책 정규화
                policy = policy / np.sum(policy)
                game_policies.append(policy)

                # 행동 수행
                state.make_move(action)

            # 게임 결과
            winner = state.check_winner()

            # 승자에 따른 가치 설정
            for i in range(len(game_states)):
                player = 1 if i % 2 == 0 else -1
                value = 0 if winner == 0 else 1 if winner == player else -1
                game_values.append(value)

            # 훈련 데이터 저장
            for state, policy, value in zip(game_states, game_policies, game_values):
                self.buffer.add(state, policy, value)

    
    def train(self, batch_size=1024, epochs=10):
        # 버퍼에서 데이터를 샘플링하여 네트워크 훈련
        if len(self.buffer) < batch_size:
            print(f"Not enough data in buffer ({len(self.buffer)} < {batch_size})")
            return
        
        self.network.model.train()

        for epoch in range(epochs):
            # 데이터 샘플링
            batch = self.buffer.sample(batch_size)
            states, policies, values = zip(*batch)

            # 텐서로 변환
            state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
            policy_tensor = torch.FloatTensor(np.array(policies)).to(self.device)
            value_tensor = torch.FloatTensor(np.array(values)).reshape(-1, 1).to(self.device)

            # 순전파
            policy_logits, value_pred = self.network.model(state_tensor)

            # 손실 계산
            policy_loss = F.cross_entropy(policy_logits, policy_tensor)
            value_loss = F.mse_loss(value_pred, value_tensor)
            total_loss = policy_loss + value_loss

            # 역전파
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss.item():.4f}",
                  f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")
            
    def evaluate(self, old_network, num_games=20, temperature=0.1):
        # 이전 버전과 대국하여 새 네트워크 성능 평가
        from gomoku_env import GomokuState
        from mcts import PUCT

        new_mcts = PUCT(self.network)
        old_mcts = PUCT(old_network)

        wins = 0
        for game_idx in range(num_games):
            state = GomokuState()
            players = [new_mcts, old_mcts] if game_idx % 2 == 0 else [old_mcts, new_mcts]

            player_idx = 0
            while not state.is_terminal():
                action = players[player_idx].get_action(state, temperature)
                state.make_move(action)
                player_idx = 1 - player_idx
            
            winner = state.check_winner()
            if (winner == 1 and game_idx % 2 == 0) or (winner == -1 and game_idx % 2 == 1):
                wins += 1

        win_rate = wins/num_games
        print(f"New network win rate: {win_rate:.2f}")
        return win_rate


        