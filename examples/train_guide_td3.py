import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

def main():
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=150)
    # model = TD3(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="logs/pacman_guide"
    # )
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
    )
    model.learn(
        total_timesteps=1e5,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacman_guide_DQN"
        ),
    )


if __name__ == "__main__":
    main()
