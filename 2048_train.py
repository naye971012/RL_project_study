from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.dqn.policies import CnnPolicy

from game_setting import game_2048
import tensorboard

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

env = game_2048()
eval_env = game_2048()

policy_kwargs = dict(
    net_arch = [128]
)

model = DQN(MlpPolicy, env,
            verbose=1,
            learning_rate=1e-4,         # learning rate
            buffer_size=1000000,         # replay buffer size
            learning_starts=50000,      # 학습 전에 랜덤하게 행동함으로써 정보를 모을 state 수
            batch_size=32, # batch size
            tau=0.99,                    # target network update rate
            target_update_interval=20000,   # target network를 업데이트하는 주기
            gamma=0.99,                 # discount factor
            exploration_fraction=0.3,       # 전체 학습 time step 중 exploration을 하는 비율
            exploration_initial_eps=0.95,    # 초기 exploration 비율
            exploration_final_eps=0.05,     # 최종적으로 exploration을 하는 비율
            train_freq=1,                   # 학습 주기 => n step 또는 n episode 마다 학습
            gradient_steps=1,               # 학습 한 번 당 gradient를 업데이트하는 횟수
            policy_kwargs=policy_kwargs,         # policy network의 구조
            tensorboard_log="./logs/tensorboard/")

print("model_structure")
print(model.policy)
input("Press Enter to continue...")

model.learn(total_timesteps=400000,     # 학습할 총 time step
            log_interval=100,            # log를 출력하는 주기
            eval_env=eval_env,          # evaluation 환경
            eval_freq=500,              # evaluation 주기 (n step)
            n_eval_episodes=10,         # evaluation 시 평가할 episode 수
            tb_log_name=f'snake_dqn_{datetime.now().strftime("%Y%m%d_%H%M")}',
            eval_log_path='./logs/model/')
