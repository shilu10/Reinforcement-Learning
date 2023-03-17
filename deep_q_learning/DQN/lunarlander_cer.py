# in cer paper, they did'nt used decaying epsilon greedy.
import gymnasium as gym
import time
import signal
import time
import sys
import pickle

env = make_env("LunarLander-v2", "videos/", 50)
action_space = [_ for _ in range(env.action_space.n)]
record = True


trainer_params = {
    "noe": 650, 
    "max_steps": 10000,
    "max_eps": 0.1,
    "min_eps": 0.1,
    "eps_decay_rate": 1e-5,
    "eps": 0.1,
    "action_space": action_space,
    "is_tg": True,
    "tg_bot_freq_epi": 10,
    "record": record,
    "gamma": 0.99, 
    "lr": 0.0001, 
    "input_dims": env.observation_space.shape,
    "mem_size" : 30000,
    "batch_size" : 32,
    "replace" : 1000,
    "algo" : "DQN",
    "env_name" : "lunarlander",
    "n_actions" : len(action_space),
    "chkpt_dir": "tmp/dqn/",
    "actions": action_space,
    "target_score": 230,
    "tau": 0.01,
    "target_update": 230,
    "cer": True,
    "video_prefix": "dqn_cer"
}

    
if __name__ == "__main__": 
    
    try: 
        manage_memory()
       
        trainer = Trainer(env, trainer_params)
        episode_rewards, epsilon_history, avg_rewards, best_reward = trainer.train_rl_model()
        
        with open("dqn_cer_episode_rewards.obj", "wb") as f: 
            pickle.dump(episode_rewards, f)
        
        with open("dqn_cer_epsilon_history.obj", "wb") as f: 
            pickle.dump(epsilon_history, f)
        
        with open("dqn_cer_avg_rewards.obj", "wb") as f: 
            pickle.dump(avg_rewards, f)
            
        plot_learning_curve(episode_rewards, epsilon_history, "vanila_dqn_cer")

        model_path = "models/lunarlander_DQN_q_value_cer/"
        evaluator = Eval(env, action_space, model_path, "cer_dqn_lunarlander", 10)
        evaluator.test()
        
        
    except Exception as error: 
        raise error
        