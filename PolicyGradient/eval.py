import random 
import imageio
from policy import *
from video_recorder import RecordVideo
import tensorflow as tf 
from utils import *


class Eval: 

    def __init__(self, env_id, model_path, video_prefix, number_of_episode=50):
        self.env = make_env(env_id) 
        self.model = tf.keras.models.load_model(model_path)
        self.recorder = RecordVideo(video_prefix, 'test_videos/', 15)
        self.number_of_episode = number_of_episode
        self.action_space = action_space
        
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
                action = get_action(state, self.model)
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


video_prefix = "policy_gradient"
model_path = "models/cartpole_reinforce/"

evaluator = Eval("CartPole-v1", model_path, video_prefix, 50)
rewards, steps = evaluator.test()

print(rewards, steps)