import random 
import imageio
from policy import greedy_policy
from video_recorder import RecordVideo
import tensorflow as tf 


class Eval: 

    def __init__(self, env, action_space, model_path, number_of_episode=50):
        self.env = env 
        self.model = tf.keras.models.load_model(model_path)
        self.recorder = RecordVideo('dqn_lunarlander', 'test_videos/', 15)
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
                action =  greedy_policy(state, self.model, self.action_space)
                state, reward_prob, terminated, truncated, _ = env.step(action)
                done = terminated or truncated 
                reward += reward_prob
                step += 1 
                if episode % 10 == 0:
                    img = env.render()
                    recorder.add_image(img)
            
            rewards.append(reward)
            steps.append(step)
            recorder.save(1) if episode % 10 == 0
        
        return rewards, steps


