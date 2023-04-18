import numpy as np 
import gymnasium as gym
import pickle 

env = make_env("LunarLanderContinuous-v2")
out_dims = env.action_space.shape[0]
action_bound = env.action_space.high[0]
input_dims = env.observation_space.shape
std_bound = [1e-2, 1.0]
print(out_dims)
noe = 1000 
max_steps = int(1e6)
video_prefix = "ppo"
is_tg = True 
record = True
alpha = 0.0001
beta = 0.001
gamma = 0.99
lmbda = 0.94
agent_update_interval = 10

if __name__ == "__main__": 
  
    trainer = Trainer(env, input_dims, out_dims, video_prefix, is_tg, 
                         noe, max_steps, record, alpha, beta, gamma, action_bound, 
                         std_bound, agent_update_interval, lmbda)
    ep_rewards, avg_rewards = trainer.train()

    with open("ppo_episode_rewards.obj", "wb") as f: 
        pickle.dump(ep_rewards, f)

    with open("ppo_avg_rewards.obj", "wb") as f: 
        pickle.dump(avg_rewards, f)
    
    plot_learning_curve(ep_rewards, "ppo.png")   
