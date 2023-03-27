import numpy as np 
import gymnasium as gym
from utils import * 
from train import *



env = make_env("CartPole-v1")
action_space = [_ for _ in range((env.action_space.n))]
n_actions = len(action_space)
input_dims = env.observation_space.shape
noe = 1000 
print(input_dims, n_actions)
max_steps = 1000000
video_prefix = "actor_critic"
is_tg = True 
record = True
lr1 = 1e-4
lr2 = 1e-4
gamma = 0.95
chpkt = 'models/'
algo_name = "actor_critic"

if __name__ == "__main__": 
  
    trainer = Trainer(env, action_space, input_dims, n_actions, video_prefix, is_tg,
                                       noe, max_steps, record, lr1, lr2, gamma, chpkt, algo_name)
    ep_rewards, avg_rewards = trainer.train()

    with open("actor_critic_eps_rewards.obj", "wb") as f: 
        pickle.dump(ep_rewards, f)

    with open("actor_critic_avg_rewards.obj", "wb") as f: 
        pickle.dump(avg_rewards, f)
    
    plot_learning_curve(ep_rewards, "actor_critic.png")