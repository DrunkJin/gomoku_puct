import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        return F.relu(x)

class GomokuNet(nn.Module):
    def __init__(self, board_size = 15, num_channels = 128, num_res_blocks = 10):
        super().__init__()
        self.board_size = board_size

        # 입력 처리
        self.conv_input = nn.Sequential(
            nn.Conv2d(3, num_channels, 3, padding = 1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU()
        )

        # Residual Blocks
        self.res_blocks = nn.ModuleList([
            ResBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # 정책 헤드
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, board_size * board_size)
        )

        # 가치 헤드
        self.value_head = nn.Sequential(
            nn.Conv2d(num_channels, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(),
            nn.Linear(256,  1),
            nn.Tanh()
        )

    def forward(self, x):
        # 공통 특징 추출
        x = self.conv_input(x)
        for res_blocks in self.res_blocks:
            x = res_blocks(x)

        # 가치와 정책 출력
        policy_logits = self.policy_head(x)
        value = self.value_head(x)

        return policy_logits, value
    
    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            policy_logits, value = self(state_tensor)
            policy = F.softmax(policy_logits, dim = 1)
            return policy, value
        
class NetWrapper:
    def __init__(self, board_size = 15, device='cuda'):
        self.board_size = board_size
        self.device = device
        self.model = GomokuNet(board_size).to(device)

    def predict(self, state):
        # 단일 상태에 대한 예측
        state_tensor = torch.FloatTensor(state.get_encoded_state()).unsqueeze(0).to(self.device)
        policy, value = self.model.predict(state_tensor)

        # 정책을 2D 배열로 전환
        policy = policy.reshape(-1, self.board_size, self.board_size)

        return (policy.cpu().numpy()[0],
                value.cpu().numpy()[0][0])
    
    def save_checkpoint(self, filepath):
        torch.save(self.model.state_dict(), filepath)

    def load_checkpoint(self, filepath):
        self.model.load_state_dict(torch.load(filepath))
