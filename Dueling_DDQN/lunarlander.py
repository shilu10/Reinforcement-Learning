import gymnasium as gym
import time
import signal
import time
import sys
import pickle
from utils import *
from train import * 

env = make_env("LunarLander-v2", "videos/", 50)
action_space = [_ for _ in range(env.action_space.n)]
record = True
print( env.observation_space.shape)

trainer_params = {
    "noe": 1000, 
    "max_steps": 10000,
    "max_eps": 1,
    "min_eps": 0.1,
    "eps_decay_rate": 1e-5,
    "eps": 1,
    "action_space": action_space,
    "is_tg": True,
    "tg_bot_freq_epi": 10,
    "record": record,
    "gamma": 0.99, 
    "lr": 0.001, 
    "input_dims": env.observation_space.shape,
    "mem_size" : 100000,
    "batch_size" : 32,
    "replace" : 500,
    "algo" : "dueling_dqn",
    "env_name" : "lunarlander",
    "n_actions" : len(action_space),
    "chkpt_dir": "tmp/dueling_dqn/",
    "actions": action_space,
    "target_score": 200,
    "video_prefix": "dueling_dqn",
    "checkpoint": False
}

    
if __name__ == "__main__": 
    
    try: 
        manage_memory()
       
        trainer = Trainer(env, trainer_params)
        episode_rewards, epsilon_history, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("dueling_dqn_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("dueling_dqn_epsilon_history.obj", "wb") as f: 
            pickle.dump(epsilon_history, f)
        
        with open("dueling_dqn_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
            
        plot_learning_curve(episode_rewards, epsilon_history, "dueling_dqn")

    except Exception as error: 
        raise error
        
