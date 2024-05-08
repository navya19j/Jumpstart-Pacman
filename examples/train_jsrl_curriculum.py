import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
from jsrl import get_jsrl_algorithm

def main():
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=250)
    guide_policy = DQN.load("examples/models/pacman_guide_DQN/best_model").policy
    n = 10
    max_horizon = 250
    model = get_jsrl_algorithm(DQN)(
        "CnnPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            strategy="curriculum",
            horizons=np.arange(max_horizon, -1, -max_horizon // n,)
        ),
        verbose=1,
        tensorboard_log="logs/pacman_jsrl_curriculum"
    )
    model.learn(
        total_timesteps=1e6,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacman_jsrl_curriculum_DQN"
        ),
    )


if __name__ == "__main__":
    main()
