from arm_env import Myenv
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)

# Separate evaluation env
eval_env = Myenv()
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-300, verbose=1)

# Use deterministic actions for evaluation
eval_callback = EvalCallback(
    eval_env,
    # callback_on_new_best=callback_on_best,
    log_path="./logs/test1/A2C/",
    eval_freq=1000,
    deterministic=False,
    render=False,
    n_eval_episodes=10,
)

model = A2C("MlpPolicy", env=eval_env, verbose=1)
model.learn(500000, callback=eval_callback)
