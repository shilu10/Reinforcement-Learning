import random 
import imageio
import tensorflow as tf 
import pickle
import gymnasium as gym 
from utils import *

class Eval: 

    def __init__(self, env_id, qtable_obj_path, video_prefix, number_of_episode=50): 
        self.env = make_env(env_id) 
        self.recorder = RecordVideo(video_prefix, 'test_videos/', 15)
        self.number_of_episode = number_of_episode
        self.Qtable = pickle.load(open(qtable_obj_path, "rb"))
        
    def test(self): 
        rewards = []
        steps = []
        for episode in range(self.number_of_episode): 
            done = False
            reward = 0
            step = 0
            state = env.reset(seed=random.randint(0,500))
            if episode % 10 == 0: 
                img = env.render()
                self.recorder.add_image(img) 

            while not done:
                if type(state) == tuple:
                    state = state[0]
                action = np.argmax(self.Qtable[state][:])
                state, reward_prob, terminated, truncated, _ = env.step(action)
                done = terminated or truncated 
                reward += reward_prob
                step += 1 
                if episode % 10 == 0:
                    img = env.render()
                    self.recorder.add_image(img)
            
            rewards.append(reward)
            steps.append(step)
            self.recorder.save(episode) if episode % 10 == 0 else None
        
        return rewards, steps


obj_path = "/kaggle/input/frozenlake-object/frozenlake_q_table.obj"
evaluator = Eval("FrozenLake-v1", obj_path, "qlearning", 50)
rewards, steps = evaluator.test()
print(rewards, steps)