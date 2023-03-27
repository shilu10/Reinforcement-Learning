import random 
from policy import *
from video_recorder import *
import numpy as np 
import random 
import tensorflow as tf 

class Eval: 

    def __init__(self, env, model_path, video_prefix, number_of_episode=50):
        self.env = env 
        self.model = tf.keras.models.load_model(model_path)
        self.recorder = RecordVideo(video_prefix, 'test_videos/', 15)
        self.number_of_episode = number_of_episode
        
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
                action =  get_action(state, self.model,)
                state, reward_prob, terminated, truncated, _ = env.step(action)
                done = terminated or truncated 
                reward += reward_prob
                step += 1 
                if episode % 10 == 0:
                    img = env.render()
                    self.recorder.add_image(img)
            
            rewards.append(reward)
            steps.append(step)
            self.recorder.save(1) if episode % 10 == 0 else None 
        
        return rewards, steps

video_prefix = "'actor_critic'"
model_path = "/content/models/_actor_critic_actor_network"
evaluator = Eval(env, model_path, video_prefix, 10)
rewards, steps = evaluator.test()

print(rewards, steps)