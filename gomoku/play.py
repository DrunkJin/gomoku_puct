import argparse
import os
import torch
import numpy as np

from gomoku_env import GomokuState
from gomoku_net import NetWrapper
from mcts import PUCT

def play_game(model_path=None, board_size=15, human_first=True, num_simulations=1000):
    # 인간 vs AI 대국 진행
    # 모델 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network = NetWrapper(board_size=board_size, device=device)

    if model_path and os.path.exists(model_path):
        network.load_checkpoint(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print("No model loaded, using randomly initialized network")

    # AI 에이전트 생성
    mcts = PUCT(network, num_simulations=num_simulations, c_puct=1.5)

    # 게임 상태 초기화
    state = GomokuState(board_size=board_size)
    state.display()

    # 플레이어 설정
    players = {
        1: "Human" if human_first else "AI",
        -1: "AI" if human_first else "Human"
    }

    # 게임 진행
    while not state.is_terminal():
        current_player = players[state.current_player]
        print(f"\n{current_player}'s turn")

        if current_player == 'Human':
            # 인간 플레이어 입력
            valid_move = False
            while not valid_move:
                try:
                    row = int(input(f"Enter row (0-{board_size-1}): "))
                    col = int(input(f"Enter column (0-{board_size-1}): "))
                    action = (row, col)

                    if 0 <= row < board_size and 0 <= col < board_size and state.board[row][col] ==0:
                        valid_move = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter valid numbers.")

        else:
            # AI 행동 선택
            print("AI is thinking...")
            action = mcts.get_action(state, temperature=0.1)
            print(f"AI places stone at: {action}")

        # 돌 놓기
        state.make_move(action)
        state.display()

        # 승자 확인
        winner = state.check_winner()
        if winner != 0:
            winner_name = player[winner]
            print(f"\nGame Over! {winner_name} wins!")
            break

        if len(state.get_legal_actions()) == 0:
            print("\nGame Over! It's a draw!")
            break

def main():
    parser = argparse.ArgumentParse(description="Play Gomoku against AI")
    parser.add.argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add.argument('--board_size', type=int, default=15, help="Board Size")
    parser.add.argument('--ai_first', action="store_true", help='AI plays first')
    parser.add.argument('--simulations', type=int, default=1000, help="Number of MCTS simulations")

    args = parser.parse_args()

    play_game(
        model_path = args.model_path,
        board_size = args.board_size,
        human_first=not args.ai_frist,
        num_simulations=args.simulations
    )

if __name__ == "__main__":
    main()

    