## main.py 실행 방법

`main.py`는 강화 학습을 사용하여 보드 게임 AI를 훈련하는 스크립트입니다. 실행 시 다양한 설정 값을 조정할 수 있도록 여러 개의 명령줄 인수를 제공합니다.

## 명령줄 인수 설명
| 인수                 | 타입      | 기본값          | 설명             |
| ------------------ | ------- | ------------ | -------------- |
| `--board_size`     | `int`   | `15`         | 보드 크기          |
| `--save_dir`       | `str`   | "./models"  | 모델 저장 디렉터리     |
| `--iterations`     | `int`   | `5`          | 학습 반복 횟수       |
| `--games_per_iter` | `int`   | `10`         | 반복당 자체 대국 수    |
| `--mcts_sims`      | `int`   | `800`        | MCTS 탐색 횟수     |
| `--c_puct`         | `float` | `1.0`        | PUCT 탐색 상수     |
| `--temp_cutoff`    | `int`   | `10`         | 온도 조정 임계값      |
| `--batch_size`     | `int`   | `512`        | 학습 배치 크기       |
| `--epochs`         | `int`   | `5`          | 반복당 학습 epoch 수 |
| `--learning_rate`  | `float` | `0.001`      | 학습률            |
| `--eval_games`     | `int`   | `10`         | 평가 대국 수        |


## 실행 예제

아래 명령어를 사용하여 `main.py`를 실행할 수 있습니다.

```bash
python main.py --board_size 19 --save_dir "./trained_models" --iterations 10 --games_per_iter 20 --mcts_sims 1000 --c_puct 1.5 --temp_cutoff 15 --batch_size 1024 --epochs 10 --learning_rate 0.0005 --eval_games 20
```

위 명령어는 다음과 같은 설정으로 실행됩니다:
- 보드 크기: 19x19
- 모델 저장 디렉터리: ./trained_models
- 학습 반복 횟수: 10
- 반복당 자체 대국 수: 20
- MCTS 탐색 횟수: 1000
- PUCT 탐색 상수: 1.5
- 온도 조정 임계값: 15
- 배치 크기: 1024
- 학습 epoch 수: 10
- 학습률: 0.0005
- 평가 대국 수: 20



## play.py 실행 방법

`play.py`를 사용하면 AI와 직접 오목을 둘 수 있습니다. 실행 시 여러 옵션을 설정할 수 있습니다.

## 명령줄 인수 설명
| 인수               | 타입      | 기본값 | 설명                        |
| ---------------- | ------- | ----- | ------------------------- |
| `--model_path`   | `str`   | `None` | 모델 체크포인트 경로             |
| `--board_size`   | `int`   | `15`   | 보드 크기                     |
| `--ai_first`     | `flag`  | `False` | AI가 먼저 두도록 설정            |
| `--simulations`  | `int`   | `1000`  | MCTS 탐색 횟수                 |

## 실행 예제

아래 명령어를 사용하여 `play.py`를 실행할 수 있습니다.

```bash
python play.py --model_path "./trained_models/best_model.pth" --board_size 15 --ai_first --simulations 1500
```

위 명령어는 다음과 같은 설정으로 실행됩니다:
- 모델 경로: ./trained_models/best_model.pth
- 보드 크기: 15x15
- AI가 먼저 둠
- MCTS 탐색 횟수: 1500

AI와 직접 대국을 진행하면서 성능을 평가할 수 있습니다.


