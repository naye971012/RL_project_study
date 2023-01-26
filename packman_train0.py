from datetime import datetime
from stable_baselines3 import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.dqn.policies import CnnPolicy

from packman_env0 import packman
import tensorboard

#observation_space가 커서 기본 training timestep을 50만으로 하였습니다

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn
import gym

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        #print(observation_space.shape) 1*11*13
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 2, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(2, 4, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
         #   nn.Conv2d(4, 4, kernel_size=2, stride=1, padding=0),
         #   nn.ReLU(),
            nn.Flatten(),
        )
        #print(observation_space.sample()[None].shape) # 1,11,13
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

env = packman()
eval_env = packman()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    net_arch = [128]
)

model = DQN(CnnPolicy, env,
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

model.learn(total_timesteps=500000,     # 학습할 총 time step
            log_interval=100,            # log를 출력하는 주기
            eval_env=eval_env,          # evaluation 환경
            eval_freq=500,              # evaluation 주기 (n step)
            n_eval_episodes=10,         # evaluation 시 평가할 episode 수
            tb_log_name=f'snake_dqn_{datetime.now().strftime("%Y%m%d_%H%M")}',
            eval_log_path='./logs/model/')
