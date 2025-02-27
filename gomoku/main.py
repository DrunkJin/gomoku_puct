import argparse
import os
import torch
import shutil

from gomoku_net import NetWrapper
from mcts import PUCT
from train import Trainer

def train_iteration(args):
    # 학습 반복 수행
    # 모델 및 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 저장 경로 생성
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 네트워크 초기화
    network = NetWrapper(board_size=args.board_size, device=device)

    # 기존 모델 로드 (있는 경우)
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    current_model_path = os.path.join(args.save_dir, "current_model.pth")

    if os.path.exists(current_model_path):
        network.load_checkpoint(current_model_path)
        print(f"Loaded existing model from {current_model_path}")

    # MCTS 에이전트 및 트레이너 초기화
    mcts = PUCT(network, num_simulations=args.mcts_sims, c_puct=args.c_puct)
    trainer = Trainer(network, device=device, lr=args.learning_rate)

    # 학습 반복
    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")

        # 자가 대국 데이터 생성
        trainer.generate_self_play_data(
            mcts, 
            num_games=args.games_per_iter,
            temperature_cutoff=args.temp_cutoff
        )

        # 네트워크 훈련
        trainer.train(batch_size= args.batch_size, epochs=args.epochs)

        # 현재 모델 저장
        network.save_checkpoint(current_model_path)
        print(f"Saved current model to {current_model_path}")

        # 이전 최고 모델과 비교
        if os.path.exists(best_model_path):
            print("Evaluationg against best model...")
            best_network = NetWrapper(board_size=args.board_size, device=device)
            best_network.load_checkpoint(best_model_path)

            win_rate = trainer.evaluate(
                best_network,
                num_games=args.eval_games,
                temperature=0.1
            )


            # 더 나은 성능이면 최고 모델 업데이트
            if win_rate > 0.55:
                shutil.copy(current_model_path, best_model_path)
                print(f"New best model with win rate: {win_rate:.2f}")
            else:
                print(f"Current model not better. Win rate: {win_rate:.2f}")
        else:
            # 첫 모델이면 바로 최고 모델로 저장
            shutil.copy(current_model_path, best_model_path)
            print(f"Saved initial best model to {best_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train Gomoku AI")

    # 기본 설정
    parser.add_argument('--board_size', type=int, default=15, help="Board Size")
    parser.add_argument('--save_dir', type=str, default="./models", help="Model save directory")
    parser.add_argument('--iterations', type=int, default=5, help="Number of training iterations")

    # 자가 대국 설정
    parser.add_argument('--games_per_iter', type=int, default=10, help="Self-play games per iteration")
    parser.add_argument('--mcts_sims', type=int, default=800, help="MCTS simulations per move")
    parser.add_argument('--c_puct', type=float, default=1.0, help="PUCT exploration constant")
    parser.add_argument('--temp_cutoff', type=int, default=10, help="Temperature cutoff move")
    
    # 훈련 설정
    parser.add_argument('--batch_size', type=int, default=512, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=5, help="Training epochs per iteration")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate")

    # 평가 설정
    parser.add_argument('--eval_games', type=int, default=10, help="Evaluation games")

    args = parser.parse_args()
    train_iteration(args)

if __name__ == '__main__':
    main()


