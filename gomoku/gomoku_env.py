import numpy as np

class GomokuState:
    def __init__(self, board_size = 15):
        self.board_size = board_size
        self.board = np.zeros((board_size, board_size), dtype = np.int8)
        self.current_player = 1 # 1: 흑, -1:  백
        self.last_move = None
        self.move_count = 0

    def copy(self):
        new_state = GomokuState(self.board_size)
        new_state.board = self.board.copy()
        new_state.current_player = self.current_player
        new_state.last_move = self.last_move
        new_state.move_count = self.move_count
        return new_state

    def get_legal_actions(self):
        legal_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal_moves.append((i, j))
        return legal_moves
    
    def make_move(self, action):
        row, col = action
        if self.board[row][col] != 0:
            return False

        self.board[row][col] = self.current_player
        self.last_move = action
        self.move_count += 1
        self.current_player *= -1
        return True
    
    def check_winner(self):
        """
        (1,0): 수평 방향 (→)
        (0,1): 수직 방향 (↓)
        (1,1): 대각선 방향 (↘)
        (1,-1): 역대각선 방향 (↗)
        """
        if self.last_move is None:
            return 0
        
        row, col = self.last_move
        player = self.board[row][col] * -1  # 이전 플레이어

        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            count = 1
            # 정방향 확인
            """
            [●] [●] [●] [●] [○]  # 현재 위치에서 오른쪽으로 확인
             ^
            현재
            """
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r, c = r + dr, c + dc

            # 역방향 확인
            """
            [○] [●] [●] [●] [●] [●]  # 현재 위치에서 왼쪽으로 확인
                     ^
                    현재
            """
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r][c] == player:
                count += 1
                r, c = r - dr, c - dc
            
            if count >= 5:
                return player
        return 0
    
    def is_terminal(self):
        if self.check_winner() != 0:
            return True
        return len(self.get_legal_actions()) == 0
    
    def get_reward(self):
        winner = self.check_winner()
        if winner == 0:
            return 0
        return winner * self.current_player * -1
    
    def get_state_planes(self):
        """
        신경망 입력용 상태 평면 생성
        3개 채널: 흑돌, 백돌, 현재 플레이어 차례
        """
        state_planes = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        state_planes[0] = (self.board == 1).astype(np.float32)
        state_planes[1] = (self.board == -1).astype(np.float32)
        state_planes[2] = np.full_like(self.board, self.current_player, dtype=np.float32)
        return state_planes
    
    def get_encoded_state(self):
        # 훈련 데이터용 상태 인코딩
        return self.get_state_planes()
    
    def display(self):
        # 콘솔용 보드 출력
        symbols = {0: '.', 1: '●', -1: '○'}
        
        print('   ' + ' '.join([f'{i:2}' for i in range(self.board_size)]))
        
        # 보드 출력 - 점 사이 간격 늘림
        for i in range(self.board_size):
            print(f'{i:2} ', end='')
            for j in range(self.board_size):
                print(f'{symbols[self.board[i][j]]}  ', end='')  # 점 뒤에 공백 두 개로 늘림
            print()





