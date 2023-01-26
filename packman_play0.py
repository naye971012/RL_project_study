import argparse
import os
import platform
import time

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from packman_env0 import packman


pwd = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best_model')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--mode', type=str, default='human')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_known_args()[0]

    platform_name = platform.system()
    print(platform_name)
    print(args)

    #model_path = os.path.join(pwd, 'logs/model', args.model)
    model_path = "logs\\model\\best_model"
    print(model_path)
    model = DQN.load(model_path)

    try:
        env = packman()
    except Exception as e:
        print(e)
        raise Exception('혹시 Environment에 새로운 argument를 추가했다면 여기에도 추가해 줘야 합니다!!')

    obs = env.reset()
    cum_reward = 0
    done, info = False, {}
    i = 0

    env.render()#mode=args.mode)

    while not done and i < args.max_steps:
        time.sleep(0.2)
        if platform_name == 'Windows':
            os.system('cls')
        else:
            os.system('clear')
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        cum_reward += reward
        env.render() #mode=args.mode)
        i += 1
        print(f'reward: {reward:.5e}, cum_reward: {cum_reward:.5e}, coin: {env.eat_num}')
    print(f"info: {info}")

    if args.eval:
        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
        print(f'Evaluation:')
        print(f'====================')
        print(f"mean_reward: {mean_reward}, std_reward: {std_reward}")


if __name__ == "__main__":
    main()
