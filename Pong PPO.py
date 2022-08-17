from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)

model = PPO.load("ppo_pong")

evaluate_policy(model,env,n_eval_episodes=10,render=True)

##
import time

obs = env.reset()
for i in range(10000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    time.sleep(0.03)
    env.render()

