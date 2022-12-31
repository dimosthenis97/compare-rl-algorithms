from pendubot_env import Pendubot
import os
from stable_baselines3 import DDPG

from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# Separate evaluation env
eval_env = Pendubot()
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    # callback_on_new_best=callback_on_best,
    log_path="./logs/test1/DDPG/",
    eval_freq=1000,
    deterministic=True,
    render=False,
    n_eval_episodes=10,
)

model = DDPG("MlpPolicy", env=eval_env, verbose=1)
model.learn(500000, callback=eval_callback)
