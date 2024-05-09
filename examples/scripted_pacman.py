import gymnasium as gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import A2C
from typing import Tuple, Any, Optional
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)
from huggingface_sb3 import EnvironmentName
from collections import OrderedDict
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from gymnasium import spaces

class ScriptedPacManPolicy(ActorCriticPolicy):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, *args, **kwargs):
        # Initialize the base policy
        super(ScriptedPacManPolicy, self).__init__(observation_space, action_space, *args, **kwargs,
                                                   net_arch=[dict(pi=[64, 64], vf=[64, 64])])

    def predict(self, observation: np.ndarray, state=None, episode_start=False, deterministic=False) -> Tuple[np.ndarray, Any]:
        """
        Predict the action for the given observation using a scripted policy.

        Parameters:
        - observation: The current state observation from the environment.
        - state: Optional state information (if any).
        - episode_start: Whether the current step is the beginning of a new episode.
        - deterministic: Whether the prediction should be deterministic.

        Returns:
        - action: The chosen action based on the observation.
        - state: Any state information (if applicable).
        """
        # observation = observation[0]
        # print(observation[2])
        print(observation.shape)
        observation = observation.reshape([observation.shape[0], observation.shape[2], observation.shape[3]])
        agent_x, agent_y = observation[:2]  # Agent's position
        print(agent_x.shape, agent_y.shape)
        # print(agent_x, agent_y)
        num_ghosts = int(observation.shape[1] - 2) // 2
        ghost_positions = observation[3:3 + num_ghosts * 2]
        remaining_observations = observation[3 + num_ghosts * 2:]
        pellet_positions = remaining_observations[:len(remaining_observations) // 2]
        power_pellet_positions = remaining_observations[len(remaining_observations) // 2:]

        # Define the possible actions
        actions = {
            0: 'NOOP',
            1: 'UP',
            2: 'RIGHT',
            3: 'LEFT',
            4: 'DOWN',
            5: 'UPRIGHT',
            6: 'UPLEFT',
            7: 'DOWNRIGHT',
            8: 'DOWNLEFT'
        }
        
        # Initialize variables for action selection
        safe_directions = set(range(9))  # All possible actions are initially safe
        
        # Avoid ghosts
        for i in range(0, len(ghost_positions), 2):
            ghost_x, ghost_y = ghost_positions[i], ghost_positions[i + 1]
            distance_to_ghost = abs(agent_x - ghost_x) + abs(agent_y - ghost_y)
            print(distance_to_ghost)
            
            # Remove unsafe actions
            if distance_to_ghost <= 1:  # Adjust threshold as needed
                if agent_x == ghost_x:
                    if ghost_y > agent_y:
                        safe_directions.discard(4)  # DOWN
                    else:
                        safe_directions.discard(1)  # UP
                elif agent_y == ghost_y:
                    if ghost_x > agent_x:
                        safe_directions.discard(2)  # RIGHT
                    else:
                        safe_directions.discard(3)  # LEFT

        # Move towards the nearest pellet
        nearest_pellet_distance = float('inf')
        nearest_pellet_direction = None
        
        for i in range(0, len(pellet_positions), 2):
            pellet_x, pellet_y = pellet_positions[i], pellet_positions[i + 1]
            distance = abs(agent_x - pellet_x) + abs(agent_y - pellet_y)
            
            if distance < nearest_pellet_distance:
                nearest_pellet_distance = distance
                if agent_x < pellet_x:
                    if agent_y < pellet_y:
                        nearest_pellet_direction = 7  # DOWNRIGHT
                    elif agent_y > pellet_y:
                        nearest_pellet_direction = 5  # UPRIGHT
                    else:
                        nearest_pellet_direction = 2  # RIGHT
                elif agent_x > pellet_x:
                    if agent_y < pellet_y:
                        nearest_pellet_direction = 8  # DOWNLEFT
                    elif agent_y > pellet_y:
                        nearest_pellet_direction = 6  # UPLEFT
                    else:
                        nearest_pellet_direction = 3  # LEFT
                else:
                    if agent_y < pellet_y:
                        nearest_pellet_direction = 4  # DOWN
                    else:
                        nearest_pellet_direction = 1  # UP
        
        # Choose the final action
        if nearest_pellet_direction is not None and nearest_pellet_direction in safe_directions:
            action = nearest_pellet_direction
        else:
            # If no clear path to a pellet, choose a random safe direction
            action = np.random.choice(list(safe_directions))

        # Return the action and state
        return np.array([action]), state

# Create and wrap the environment
env = gym.make('ALE/MsPacman-v5')  # Replace with your specific Pac-Man environment
env_name = EnvironmentName("ALE/MsPacman-v5")
env_kwargs = {}
normalize = False
seed = 0
frame_stack = 1
vec_env_type = "dummy"
vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]

spec = gym.spec(env_name.gym_id)

def make_env(**kwargs) -> gym.Env:
    return spec.make(**kwargs)

# Make the environment
env = make_atari_env(
    make_env,
    n_envs=8,
    seed=seed,
    vec_env_cls=vec_env_class,
    vec_env_kwargs=env_kwargs,
)
if frame_stack:
    env = VecFrameStack(env, n_stack=frame_stack)

if not is_vecenv_wrapped(env, VecTransposeImage):
    wrap_with_vectranspose = False
    if isinstance(env.observation_space, spaces.Dict):
        # If even one of the keys is an image-space in need of transpose, apply transpose
        # If the image spaces are not consistent (for instance, one is channel first,
        # the other channel last); VecTransposeImage will throw an error
        for space in env.observation_space.spaces.values():
            wrap_with_vectranspose = wrap_with_vectranspose or (
                is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
            )
    else:
        wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
            env.observation_space  # type: ignore[arg-type]
        )

    if wrap_with_vectranspose:
        print("Wrapping the env in a VecTransposeImage.")
        env = VecTransposeImage(env)

# Create an instance of the A2C model using your custom policy
model = A2C(ScriptedPacManPolicy, env, verbose=1)

# Train the agent
model.learn(total_timesteps=10,
            log_interval=10,
            progress_bar=True,
            callback=EvalCallback(
                env, 
                n_eval_episodes=10, 
                best_model_save_path="a2c-pacman"))

# test the agent for mean reward
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# # Save the model
# model.save("a2c-pacman")

# # Load the model with the custom policy
# loaded_model = A2C.load("a2c-pacman", policy=ScriptedPacManPolicy)