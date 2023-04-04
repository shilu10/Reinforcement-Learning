import gymnasium as gym
import time
import signal
import time
import sys
import pickle
import os 

env = make_env("MountainCarContinuous-v0", "videos/", 50)
record = False
gamma = 0.9
alpha = 0.001
beta = 0.002
batch_size = 64
tau = 0.01 
soft_update = True 
noe = 500
max_steps = 100000
is_tg = True 
tg_bot_freq_epi = 20
record = True 
mem_size = 5000000
noise = 0.5
    
    
if not os.path.exists("videos"): 
    os.mkdir("videos")

if not os.path.exists("test_videos"):
    os.mkdir("test_videos")


if __name__ == "__main__": 
    
    try: 
        manage_memory()
        trainer = Trainer(env, gamma, alpha, beta, batch_size, tau,
                          soft_update, noe, max_steps, is_tg,
                          tg_bot_freq_epi, record, mem_size, noise
                )
        episode_rewards, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("ddpg_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("ddpg_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
        
        x = [i+1 for i in range(len(episode_rewards))]
        plot_learning_curve(x, episode_rewards, "ddpg_con_mountain_car")

       # model_path = "models/lunarlander_DQN_q_value/"

        #evaluator = Eval(env, action_space, model_path, "vanilla_dqn_lunarlander", 10)
        #evaluator.test()
        
    except Exception as error: 
        raise error
        
   # eval_model(env, "keras model", "videos/", fps=10)
