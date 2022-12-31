from arm_env import Myenv
import gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# Separate evaluation env
eval_env = Myenv()
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

# Use deterministic actions for evaluation
# eval_callback = EvalCallback(
#     eval_env,
#     # callback_on_new_best=callback_on_best,
#     log_path="./logs/test1/PPO/",
#     eval_freq=1000,
#     deterministic=False,
#     render=False,
#     n_eval_episodes=10,
# )

model = PPO("MlpPolicy", env=eval_env, verbose=1)
model.learn(
    500000,
)
