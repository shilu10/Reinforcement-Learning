from train import * 
from eval import *
import gymnasium as gym
import time
import signal
import time
import sys
import pickle
import os 


if not os.path.exists("videos"): 
    os.mkdir("videos")

if not os.path.exists("test_videos"):
    os.mkdir("test_videos")

env = make_env("LunarLander-v2", "videos/", 50)
action_space = [_ for _ in range(env.action_space.n)]
record = True


trainer_params = {
    "noe": 650, 
    "max_steps": 1000,
    "max_eps": 1,
    "min_eps": 0.1,
    "eps_decay_rate": 1e-5,
    "eps": 1,
    "action_space": action_space,
    "is_tg": True,
    "tg_bot_freq_epi": 10,
    "record": record,
    "gamma": 0.99, 
    "lr": 0.0001, 
    "input_dims": env.observation_space.shape,
    "mem_size" : 20000,
    "batch_size" : 32,
    "replace" : 1000,
    "algo" : "DQN",
    "env_name" : "lunarlander",
    "n_actions" : len(action_space),
    "chkpt_dir": "tmp/dqn/",
    "actions": action_space,
    "target_score": 230,
    "cer": False,
    "video_prefix": "dqn"
}

    
if __name__ == "__main__": 
    
    try: 
        manage_memory()
       
        trainer = Trainer(env, trainer_params)
        episode_rewards, epsilon_history, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("dqn_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("dqn_epsilon_history.obj", "wb") as f: 
            pickle.dump(epsilon_history, f)
        
        with open("dqn_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
            
        plot_learning_curve(episode_rewards, epsilon_history, "plot_file")

        evaluator = Eval(env, "models/lunarlander_DQN_q_value/", 10)
        evaluator.test()
        
    except Exception as error: 
        raise error
        
   # eval_model(env, "keras model", "videos/", fps=10)
