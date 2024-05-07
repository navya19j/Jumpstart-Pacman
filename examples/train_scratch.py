# import gymnasium as gym
# import gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym
import sys
sys.modules["gym"] = gym

def main():
    env = gym.make("ALE/MsPacman-v5")
    # any baseline model can be used here - Q learning
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
    )
    # model = TD3(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="logs/pacman_scratch"
    # )
    model.learn(
        total_timesteps=1e5,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacman_scratch_DQN"
        ),
    )

    model.save("examples/models/pacman_scratch_DQN")


if __name__ == "__main__":
    main()
