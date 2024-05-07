import gym
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import EvalCallback
from jsrl import get_jsrl_algorithm

def main():
    env = gym.make("ALE/MsPacman-v5",max_episode_steps=150)
    guide_policy = TD3.load("examples/models/pacman_guide_TD3/best_model").policy
    max_horizon = 60
    model = get_jsrl_algorithm(TD3)(
        "MultiInputPolicy",
        env,
        policy_kwargs=dict(
            guide_policy=guide_policy,
            max_horizon=max_horizon,
            strategy="random"
        ),
        verbose=1,
        tensorboard_log="logs/pacman_jsrl_random"
    )
    model.learn(
        total_timesteps=1e5,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacman_jsrl_random_TD3"
        ),
    )


if __name__ == "__main__":
    main()
