# import gymnasium as gym
# import gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import sys
sys.modules["gym"] = gym

def main():
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=150)
    model = TD3(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log="logs/pointmaze_scratch"
    )
    model.learn(
        total_timesteps=1e6,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pointmaze_scratch_TD3"
        ),
    )


if __name__ == "__main__":
    main()
